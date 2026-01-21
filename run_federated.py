from __future__ import annotations

import argparse
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv

# Make project root importable
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Env.road_graph_builder import RoadGraphBuilder
from FL_PPO.fed_ppo_fedavg import FederatedPPOCoordinator, SB3PPOClient
from FL_PPO.rl_utils.custom_model import CustomTSCModel
from FL_PPO.rl_utils.make_tsc_env import make_env
from FL_PPO.rl_utils.sb3_utils import linear_schedule


def build_client(
    tls_id: str,
    tls_action_type: str,
    phase_num: int,
    sumo_cfg: str,
    net_file: str,
    num_seconds: int,
    n_envs: int,
    log_root: Path,
    tb_root: Path,
    device: str,
    use_gui: bool = False,
    label_prefix: str = "fl",
) -> SB3PPOClient:
    """
    Create a PPO client with its own VecNormalize + SubprocVecEnv.
    Boundary for FL: only model.policy.state_dict is synchronized; VecNormalize stays local.
    """
    log_root.mkdir(parents=True, exist_ok=True)
    tb_root.mkdir(parents=True, exist_ok=True)

    params = {
        "tls_id": tls_id,
        "num_seconds": num_seconds,
        "tls_action_type": tls_action_type,
        "phase_num": phase_num,
        "sumo_cfg": sumo_cfg,
        "net_file": net_file,
        "use_gui": use_gui,
        "log_file": str(log_root),
        "label_prefix": label_prefix,
    }
    env_fns = [make_env(env_index=f"{tls_id}-{i}", **params) for i in range(n_envs)]
    if n_envs == 1:
        vec_env = DummyVecEnv(env_fns)
    else:
        vec_env = SubprocVecEnv(env_fns)
    vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True)

    policy_kwargs = dict(
        features_extractor_class=CustomTSCModel,
        features_extractor_kwargs=dict(features_dim=16),
    )
    model = PPO(
        "MlpPolicy",
        vec_env,
        batch_size=64,
        n_steps=300,
        n_epochs=5,
        learning_rate=linear_schedule(1e-3),
        verbose=1,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(tb_root),
        device=device,
    )
    return SB3PPOClient(tls_id=tls_id, model=model)


def select_participants(clients: List[SB3PPOClient], rate: float) -> List[SB3PPOClient]:
    if rate >= 1.0 or len(clients) == 1:
        return list(clients)
    k = max(1, int(len(clients) * rate))
    return random.sample(clients, k)


def summarize_rewards(updates):
    rewards = [
        u.metrics.get("reward_mean")
        for u in updates
        if u.metrics.get("reward_mean") is not None and not torch.isnan(torch.tensor(u.metrics["reward_mean"]))
    ]
    if not rewards:
        return None, None, None
    return float(sum(rewards) / len(rewards)), float(min(rewards)), float(max(rewards))


def run_federated(
    env_name: str = "4nodes",
    tls_action_type: str = "next_or_not",
    tls_ids: Optional[List[str]] = None,
    rounds: int = 3,
    local_timesteps: int = 5000,
    participation_rate: float = 1.0,
    num_seconds: int = 500,
    n_envs: int = 1,
    use_gui: bool = False,
    device: Optional[str] = None,
    personalization_steps: int = 0,
    parallel_clients: bool = False,
    use_subnet_per_tls: bool = False,
    fl_alpha: float = 0.0,
    max_monitor_rows: Optional[int] = None,
) -> None:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    sumo_cfg = str(Path(f"Scenario/{env_name}/env/vehicle.sumocfg").resolve())
    net_file = str(Path(f"Scenario/{env_name}/env/{env_name}.net.xml").resolve())
    phase_num = 2 if tls_action_type == "next_or_not" else 4

    if tls_ids is None:
        graph = RoadGraphBuilder.build_from_net_file(
            net_file=net_file,
            directed=True,
            make_bidirectional=False,
            include_self_loops=False,
            max_hops_between_tls=1,
            neighbor_strategy="hop",
            neighbor_hop_k=1,
            include_self_in_neighbor=True,
        )
        tls_ids = graph.node_id_list or graph.idx_to_tls_id

    print(f"[INFO] TLS clients: {tls_ids}")
    print("[INFO] Each client trains on full SUMO net but controls only its tls_id; FedAvg syncs policies across clients.")
    log_root = Path(f"FL_PPO/result/{env_name}/{tls_action_type}/log")
    tb_root = Path(f"FL_PPO/result/{env_name}/{tls_action_type}/tensorboard")
    client_ckpt_root = Path(f"FL_PPO/result/{env_name}/{tls_action_type}/clients")
    global_ckpt_root = Path("Models")

    def count_monitor_rows(log_dir: Path) -> int:
        if not log_dir.exists():
            return 0
        total = 0
        for csv_file in log_dir.rglob("*.monitor.csv"):
            try:
                with csv_file.open() as f:
                    lines = sum(1 for _ in f)
                total += max(lines - 2, 0)  # skip monitor headers (#metadata + column names)
            except Exception:
                continue
        return total

    # Optionally use per-TLS sub-scenarios if provided under Scenario/<tls_id>/env/{vehicle.sumocfg, <tls_id>.net.xml}
    def resolve_paths_for_tls(tls: str) -> tuple[str, str]:
        if not use_subnet_per_tls:
            return sumo_cfg, net_file
        candidate_cfg = Path(f"Scenario/{tls}/env/vehicle.sumocfg").resolve()
        candidate_net = Path(f"Scenario/{tls}/env/{tls}.net.xml").resolve()
        if candidate_cfg.exists() and candidate_net.exists():
            return str(candidate_cfg), str(candidate_net)
        # Fallback to global if subnet missing
        return sumo_cfg, net_file

    clients: List[SB3PPOClient] = []
    for tls_id in tls_ids:
        tls_sumo_cfg, tls_net_file = resolve_paths_for_tls(tls_id)
        clients.append(
            build_client(
                tls_id=tls_id,
                tls_action_type=tls_action_type,
                phase_num=phase_num,
                sumo_cfg=tls_sumo_cfg,
                net_file=tls_net_file,
                num_seconds=num_seconds,
                n_envs=n_envs,
                log_root=log_root / tls_id,
                tb_root=tb_root / tls_id,
                device=device,
                use_gui=use_gui,
                label_prefix=f"fl-{tls_id}",
            )
        )

    coordinator = FederatedPPOCoordinator(alpha=fl_alpha)
    global_state = clients[0].get_policy_state()
    coordinator.broadcast(clients, global_state)

    for rnd in range(1, rounds + 1):
        participants = select_participants(clients, participation_rate)
        print(f"[ROUND {rnd}] participants: {[c.tls_id for c in participants]}")

        updates = []
        if parallel_clients:
            with ThreadPoolExecutor(max_workers=len(participants)) as executor:
                future_map = {
                    executor.submit(client.train_local, local_timesteps, CallbackList([])): client
                    for client in participants
                }
                for future in as_completed(future_map):
                    client = future_map[future]
                    update = future.result()
                    updates.append(update)
                    client.save_local(client_ckpt_root / client.tls_id, tag=f"round{rnd}")
        else:
            for client in participants:
                update = client.train_local(local_timesteps, callback=CallbackList([]))
                updates.append(update)
                # Save local policy + vecnorm per client per round
                client.save_local(client_ckpt_root / client.tls_id, tag=f"round{rnd}")

        global_state = coordinator.aggregate(updates, prev_global=global_state)
        coordinator.save_global(global_ckpt_root / f"global_round_{rnd}.pt", global_state)
        coordinator.broadcast(clients, global_state)

        mean_r, min_r, max_r = summarize_rewards(updates)
        print(f"[ROUND {rnd}] reward_mean={mean_r}, min={min_r}, max={max_r}, updates={len(updates)}")

        if personalization_steps > 0:
            for client in clients:
                client.train_local(personalization_steps, callback=CallbackList([]))

        if max_monitor_rows is not None:
            total_logs = count_monitor_rows(log_root)
            print(f"[INFO] Monitor rows so far: {total_logs}/{max_monitor_rows}")
            if total_logs >= max_monitor_rows:
                print(f"[INFO] Reached max_monitor_rows={max_monitor_rows}; stopping early.")
                break

    for client in clients:
        client.close()
    print("Federated training finished.")


def parse_args():
    parser = argparse.ArgumentParser(description="Federated PPO (FedAvg) for TSC")
    parser.add_argument("--env-name", type=str, default="1node")
    parser.add_argument("--tls-action-type", type=str, default="next_or_not")
    parser.add_argument("--tls-ids", type=str, default=None, help="Comma-separated tls ids; default: infer from net file")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--local-timesteps", type=int, default=5000)
    parser.add_argument("--participation-rate", type=float, default=1.0)
    parser.add_argument("--num-seconds", type=int, default=500)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--use-gui", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--personalization-steps", type=int, default=0)
    parser.add_argument("--parallel-clients", action="store_true", help="Train participating clients in parallel threads")
    parser.add_argument(
        "--use-subnet-per-tls",
        action="store_true",
        help="If set, look for Scenario/<tls_id>/env/vehicle.sumocfg and <tls_id>.net.xml per client; fallback to global env if missing.",
    )
    parser.add_argument("--fl-alpha", type=float, default=0.0, help="Mixing rate for previous global vs FedAvg (0 => pure FedAvg)")
    parser.add_argument("--max-monitor-rows", type=int, default=None, help="Stop early once total Monitor rows reach this number.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tls_ids = args.tls_ids.split(",") if args.tls_ids else None
    run_federated(
        env_name=args.env_name,
        tls_action_type=args.tls_action_type,
        tls_ids=tls_ids,
        rounds=args.rounds,
        local_timesteps=args.local_timesteps,
        participation_rate=args.participation_rate,
        num_seconds=args.num_seconds,
        n_envs=args.n_envs,
        use_gui=args.use_gui,
        device=args.device,
        personalization_steps=args.personalization_steps,
        parallel_clients=args.parallel_clients,
        use_subnet_per_tls=args.use_subnet_per_tls,
        fl_alpha=args.fl_alpha,
        max_monitor_rows=args.max_monitor_rows,
    )
