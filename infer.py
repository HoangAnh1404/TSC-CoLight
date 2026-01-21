import argparse
import os
from pathlib import Path
from typing import Dict, Tuple

import torch

from Models.train import QNetwork, reorder_state
from Env.make_env import make_multi_envs
from Env.road_graph_builder import GraphSpec, RoadGraphBuilder

try:
    from traci.exceptions import FatalTraCIError
except Exception:  # pragma: no cover - optional dependency
    FatalTraCIError = Exception

TLS_ACTION_TYPE = "adjust_cycle_duration"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy a trained CoLight model for SUMO simulation.")
    # parser.add_argument("--model-path", default="Models/result/q_net_final.pt", help="Path to trained QNetwork weights.")
    parser.add_argument("--model-path", default="Models/result/adjust_cycle_duration/target_net_ep2110.pt", help="Path to trained QNetwork weights.")
    parser.add_argument("--sumo-cfg", default="Scenario/test/env/vehicle.sumocfg", help="SUMO config file (.sumocfg).")
    parser.add_argument("--net-file", default="Scenario/test/env/test.net.xml", help="SUMO net file used to build the graph.")
    parser.add_argument("--trip-info", default="deploy.tripinfo.xml", help="Optional SUMO tripinfo output path.")
    parser.add_argument("--num-seconds", type=int, default=500, help="Simulation horizon in seconds.")
    parser.add_argument("--log-path", default='deploy.log.monitor.csv', help="Path to monitor log output.")
    parser.add_argument("--gui", dest="use_gui", action="store_true", help="Run SUMO with GUI.")
    parser.add_argument("--no-gui", dest="use_gui", action="store_false", help="Run SUMO headless.")
    parser.set_defaults(use_gui=False)
    return parser.parse_args()


def build_graph(net_file: str) -> Tuple[GraphSpec, list, torch.Tensor]:
    graph = RoadGraphBuilder.build_from_net_file(
        net_file=net_file,
        directed=True,
        make_bidirectional=False,
        include_self_loops=False,
        max_hops_between_tls=1,
        neighbor_strategy="hop",
        neighbor_hop_k=1,
        neighbor_top_k=None,
        include_self_in_neighbor=True,
    )
    node_id_list = graph.node_id_list or graph.idx_to_tls_id
    edge_index = graph.neighbors_edge_index(add_reverse=True, add_self_loops=True)
    if not isinstance(edge_index, torch.Tensor):
        edge_index = torch.as_tensor(edge_index, dtype=torch.long)
    return graph, node_id_list, edge_index


def prepare_env(node_id_list, sumo_cfg: str, net_file: str, trip_info: str, num_seconds: int, use_gui: bool, log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return make_multi_envs(
        tls_ids=node_id_list,
        sumo_cfg=sumo_cfg,
        num_seconds=num_seconds,
        use_gui=use_gui,
        net_file=net_file,
        trip_info=trip_info,
        tls_action_type=TLS_ACTION_TYPE,
        log_path=str(log_path),
    )


def infer_state_dims(state: Dict[str, Dict[str, torch.Tensor]]) -> Tuple[int, int]:
    sample_entry = next(iter(state.values()))
    occupancy = sample_entry.get("occupancy", sample_entry)
    phase = sample_entry.get("phase", [])
    return len(occupancy), len(phase)


def init_model(model_path: Path, occupancy_dim: int, phase_dim: int, action_dim: int, device: torch.device) -> QNetwork:
    model = QNetwork(
        occupancy_dim=occupancy_dim,
        phase_dim=phase_dim,
        hidden_dim=128,
        action_dim=action_dim,
        heads=4,
        tau=1.0,
        dropout=0.1,
    ).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def resolve_trip_info_path(trip_info_arg: str, log_path: Path) -> Path | None:
    """
    Resolve a writable trip_info path.

    - None -> None (disable tripinfo output)
    - If the provided path is not writable, fall back to log_path.with_suffix(".tripinfo.xml")
    """
    if not trip_info_arg:
        return None

    candidate = Path(trip_info_arg)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate

    fallback = Path(log_path).with_suffix(".tripinfo.xml")

    try:
        candidate.parent.mkdir(parents=True, exist_ok=True)
        if not os.access(candidate.parent, os.W_OK):
            raise PermissionError(f"No write permission to {candidate.parent}")
    except PermissionError:
        fallback.parent.mkdir(parents=True, exist_ok=True)
        print(f"Không ghi được trip_info vào {candidate}, chuyển sang {fallback}")
        return fallback

    return candidate


def rollout(
    env,
    model: QNetwork,
    graph,
    node_id_list,
    edge_index: torch.Tensor,
    phase_dim: int,
    action_dim: int,
    max_steps: int,
    initial_state,
    device: torch.device,
) -> Tuple[int, Dict[str, float]]:
    state = initial_state
    done_flags = {tls: False for tls in node_id_list}
    action_mask = torch.ones((len(node_id_list), action_dim), device=device)
    rewards = {tls: 0.0 for tls in node_id_list}

    for step_idx in range(max_steps):
        x = reorder_state(
            state,
            tls_id_to_idx=graph.tls_id_to_idx,
            node_id_list=node_id_list,
            phase_dim=phase_dim,
            device=device,
        )
        with torch.no_grad():
            q_values = model(x, edge_index)
            q_values = q_values.masked_fill(action_mask < 0.5, float("-inf"))
            actions = torch.argmax(q_values, dim=-1)

        actions_dict = {tls: int(actions[i].item()) for i, tls in enumerate(node_id_list)}
        state, reward_dict, truncated, done_dict, infos = env.step(actions_dict)
        for tls in node_id_list:
            rewards[tls] += float(reward_dict.get(tls, 0.0))
        done_flags = {tls: bool(done_dict.get(tls, False) or truncated.get(tls, False)) for tls in node_id_list}

        infos = infos or {}
        action_mask = torch.tensor(
            [infos.get(tls, {}).get("can_perform_action", True) for tls in node_id_list],
            device=device,
            dtype=torch.float32,
        ).unsqueeze(-1).expand(-1, action_dim)

        if all(done_flags.values()):
            return step_idx + 1, rewards

    return max_steps, rewards


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file trọng số: {model_path}")

    log_path = Path(args.log_path)
    trip_info_path = resolve_trip_info_path(args.trip_info, log_path)

    graph, node_id_list, edge_index = build_graph(args.net_file)
    env = prepare_env(
        node_id_list=node_id_list,
        sumo_cfg=args.sumo_cfg,
        net_file=args.net_file,
        trip_info=str(trip_info_path) if trip_info_path else None,
        num_seconds=args.num_seconds,
        use_gui=args.use_gui,
        log_path=log_path,
    )

    try:
        state, _ = env.reset()
    except FatalTraCIError as exc:
        env.close()
        if args.use_gui:
            print(f"SUMO GUI khởi động thất bại ({exc}); thử lại ở chế độ headless (--no-gui).")
            env = prepare_env(
                node_id_list=node_id_list,
                sumo_cfg=args.sumo_cfg,
                net_file=args.net_file,
                trip_info=str(trip_info_path) if trip_info_path else None,
                num_seconds=args.num_seconds,
                use_gui=False,
                log_path=log_path,
            )
            state, _ = env.reset()
        else:
            raise
    occupancy_dim, phase_dim = infer_state_dims(state)
    action_dim = 2 if TLS_ACTION_TYPE == "next_or_not" else max(1, phase_dim * 2)
    edge_index = edge_index.to(device)

    try:
        model = init_model(model_path, occupancy_dim, phase_dim, action_dim, device)
        steps, rewards = rollout(
            env=env,
            model=model,
            graph=graph,
            node_id_list=node_id_list,
            edge_index=edge_index,
            phase_dim=phase_dim,
            action_dim=action_dim,
            max_steps=args.num_seconds,
            initial_state=state,
            device=device,
        )
    finally:
        env.close()

    total_reward = sum(rewards.values())
    print(f"Hoàn thành mô phỏng: steps={steps}, total_reward={total_reward:.3f}")


if __name__ == "__main__":
    main()
