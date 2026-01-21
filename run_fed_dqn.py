from __future__ import annotations

import argparse
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

import torch

# Ensure project root importable
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Env.road_graph_builder import RoadGraphBuilder
from FedDQN.client import DQNClient, DQNConfig
from FedDQN.server import FedAvgServer
from FedDQN.utils import reorder_state


def select_participants(clients, rate: float):
    if rate >= 1.0 or len(clients) == 1:
        return list(clients)
    k = max(1, int(len(clients) * rate))
    return random.sample(clients, k)


def run_federated_dqn(
    env_name: str = "4nodes",
    tls_action_type: str = "next_or_not",
    tls_ids: Optional[List[str]] = None,
    rounds: int = 3,
    local_steps: int = 5000,
    participation_rate: float = 1.0,
    neighbor_hop_k: int = 1,
    dueling: bool = False,
    hidden_dim: int = 128,
    heads: int = 4,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    batch_size: int = 128,
    warmup_steps: int = 200,
    target_update_interval: int = 200,
    tau: float = 0.0,
    grad_clip_norm: float = 5.0,
    replay_capacity: int = 5000,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: int = 5000,
    max_steps_per_ep: int = 300,
    num_seconds: int = 500,
    use_gui: bool = False,
    device: Optional[str] = None,
    save_dir: str = "FedDQN/result",
    parallel_clients: bool = False,
    fl_alpha: float = 0.0,
) -> None:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    sumo_cfg = str(Path(f"Scenario/{env_name}/env/vehicle.sumocfg").resolve())
    net_file = str(Path(f"Scenario/{env_name}/env/{env_name}.net.xml").resolve())
    log_root = Path(f"{save_dir}/{env_name}/{tls_action_type}/log")
    client_ckpt_root = Path(f"{save_dir}/{env_name}/{tls_action_type}/clients")
    global_ckpt_root = Path(f"{save_dir}/{env_name}/{tls_action_type}/global")

    graph = RoadGraphBuilder.build_from_net_file(
        net_file=net_file,
        directed=True,
        make_bidirectional=False,
        include_self_loops=False,
        max_hops_between_tls=1,
        neighbor_strategy="hop",
        neighbor_hop_k=neighbor_hop_k,
        include_self_in_neighbor=True,
    )
    if tls_ids is None:
        tls_ids = graph.node_id_list or graph.idx_to_tls_id

    env_cfg_base = {
        "sumo_cfg": sumo_cfg,
        "net_file": net_file,
        "num_seconds": num_seconds,
        "use_gui": use_gui,
        "trip_info": None,
        "tls_action_type": tls_action_type,
        "log_dir": str(log_root),
    }
    dqn_cfg = DQNConfig(
        gamma=gamma,
        tau=tau,
        target_update_interval=target_update_interval,
        huber_delta=1.0,
        grad_clip_norm=grad_clip_norm,
        warmup_steps=warmup_steps,
        batch_size=batch_size,
    )

    clients: List[DQNClient] = []
    for tls in tls_ids:
        clients.append(
            DQNClient(
                tls_id=tls,
                graph=graph,
                env_cfg=env_cfg_base,
                dqn_config=dqn_cfg,
                hidden_dim=hidden_dim,
                heads=heads,
                learning_rate=learning_rate,
                dueling=dueling,
                neighbor_hop_k=neighbor_hop_k,
                replay_capacity=replay_capacity,
                epsilon_start=epsilon_start,
                epsilon_end=epsilon_end,
                epsilon_decay=epsilon_decay,
                max_steps_per_ep=max_steps_per_ep,
                device=device,
            )
        )

    server = FedAvgServer(alpha=fl_alpha, weight_fn=lambda u: float(max(u["steps"], 1)))
    global_state = clients[0].get_policy_state()
    server.broadcast(clients, global_state)

    for rnd in range(1, rounds + 1):
        participants = select_participants(clients, participation_rate)
        print(f"[ROUND {rnd}] participants: {[c.tls_id for c in participants]}")

        updates = []
        if parallel_clients:
            with ThreadPoolExecutor(max_workers=len(participants)) as executor:
                future_map = {executor.submit(c.train_local, local_steps): c for c in participants}
                for future in as_completed(future_map):
                    c = future_map[future]
                    res = future.result()
                    updates.append({"tls_id": c.tls_id, **res})
                    c.save_local(client_ckpt_root / c.tls_id, tag=f"round{rnd}")
        else:
            for c in participants:
                res = c.train_local(local_steps)
                updates.append({"tls_id": c.tls_id, **res})
                c.save_local(client_ckpt_root / c.tls_id, tag=f"round{rnd}")

        client_updates = []
        for u in updates:
            client_updates.append(
                type("obj", (), {
                    "tls_id": u["tls_id"],
                    "state_dict": u["state_dict"],
                    "weight": float(max(u["steps"], 1)),
                    "metrics": u.get("metrics", {}),
                })
            )

        global_state = server.aggregate(client_updates, prev_global=global_state)
        server.save_global(global_ckpt_root / f"global_round_{rnd}.pt", global_state)
        server.broadcast(clients, global_state)
        print(f"[ROUND {rnd}] aggregated; updates={len(updates)}")

    for c in clients:
        c.close()
    print("Federated DQN training finished.")


def parse_args():
    ap = argparse.ArgumentParser(description="Federated DQN with GAT for TSC")
    ap.add_argument("--env-name", type=str, default="4nodes")
    ap.add_argument("--tls-action-type", type=str, default="next_or_not")
    ap.add_argument("--tls-ids", type=str, default=None, help="Comma-separated tls ids")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--local-steps", type=int, default=5000)
    ap.add_argument("--participation-rate", type=float, default=1.0)
    ap.add_argument("--neighbor-hop-k", type=int, default=1)
    ap.add_argument("--dueling", action="store_true")
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--target-update-interval", type=int, default=200)
    ap.add_argument("--tau", type=float, default=0.0)
    ap.add_argument("--grad-clip-norm", type=float, default=5.0)
    ap.add_argument("--replay-capacity", type=int, default=5000)
    ap.add_argument("--epsilon-start", type=float, default=1.0)
    ap.add_argument("--epsilon-end", type=float, default=0.05)
    ap.add_argument("--epsilon-decay", type=int, default=5000)
    ap.add_argument("--max-steps-per-ep", type=int, default=300)
    ap.add_argument("--num-seconds", type=int, default=500)
    ap.add_argument("--use-gui", action="store_true")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--save-dir", type=str, default="FedDQN/result")
    ap.add_argument("--parallel-clients", action="store_true")
    ap.add_argument("--fl-alpha", type=float, default=0.0)
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tls_list = args.tls_ids.split(",") if args.tls_ids else None
    run_federated_dqn(
        env_name=args.env_name,
        tls_action_type=args.tls_action_type,
        tls_ids=tls_list,
        rounds=args.rounds,
        local_steps=args.local_steps,
        participation_rate=args.participation_rate,
        neighbor_hop_k=args.neighbor_hop_k,
        dueling=args.dueling,
        hidden_dim=args.hidden_dim,
        heads=args.heads,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        target_update_interval=args.target_update_interval,
        tau=args.tau,
        grad_clip_norm=args.grad_clip_norm,
        replay_capacity=args.replay_capacity,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        max_steps_per_ep=args.max_steps_per_ep,
        num_seconds=args.num_seconds,
        use_gui=args.use_gui,
        device=args.device,
        save_dir=args.save_dir,
        parallel_clients=args.parallel_clients,
        fl_alpha=args.fl_alpha,
    )
