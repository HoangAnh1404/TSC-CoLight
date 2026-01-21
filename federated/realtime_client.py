from __future__ import annotations

import argparse
import json
import os
import threading
import time
from pathlib import Path
from typing import Dict, Optional

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

# Ensure project root
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from FL_PPO.rl_utils.custom_model import CustomTSCModel
from FL_PPO.rl_utils.make_tsc_env import make_env
from FL_PPO.rl_utils.sb3_utils import linear_schedule
from FL_PPO.fed_ppo_fedavg import SB3PPOClient

REALTIME_DISABLED_MSG = "Realtime training has been disabled; use run_federated.py for standard PPO FedAvg."


def build_env(tls_id: str, params: Dict[str, str], training: bool, role: str) -> VecNormalize:
    # Ensure unique label per env to avoid TraCI collision
    params_with_label = dict(params)
    params_with_label["label_prefix"] = f"{params.get('label_prefix', 'fl')}-{tls_id}-{role}-{int(time.time() * 1000)}"
    env_fn = make_env(env_index=f"{tls_id}-{role}", **params_with_label)
    vec_env = DummyVecEnv([env_fn])
    vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True)
    vec_env.training = training
    vec_env.norm_reward = training
    return vec_env


class RealtimeClient:
    def __init__(
        self,
        client_id: str,
        params: Dict[str, str],
        updates_dir: Path,
        checkpoints_dir: Path,
        chunk_steps: int = 2000,
        alpha_merge: float = 0.1,
        gating_delta: float = 0.0,
        eval_steps: int = 200,
        device: Optional[str] = None,
    ) -> None:
        self.client_id = client_id
        self.params = params
        self.updates_dir = updates_dir
        self.ckpt_dir = checkpoints_dir
        self.chunk_steps = chunk_steps
        self.alpha_merge = alpha_merge
        self.gating_delta = gating_delta
        self.eval_steps = eval_steps
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        policy_kwargs = dict(
            features_extractor_class=CustomTSCModel,
            features_extractor_kwargs=dict(features_dim=16),
        )
        train_env = build_env(client_id, params, training=True, role="train")
        model = PPO(
            "MlpPolicy",
            train_env,
            batch_size=64,
            n_steps=300,
            n_epochs=5,
            learning_rate=linear_schedule(1e-3),
            verbose=0,
            policy_kwargs=policy_kwargs,
            device=self.device,
        )
        self.client = SB3PPOClient(tls_id=client_id, model=model)
        self.policy_train = model.policy
        # Serving policy clone (separate env, keep n_steps>1 to satisfy SB3)
        self.policy_serve = PPO(
            "MlpPolicy",
            build_env(client_id, params, training=False, role="serve"),
            batch_size=64,
            n_steps=64,
            n_epochs=1,
            learning_rate=linear_schedule(1e-3),
            verbose=0,
            policy_kwargs=policy_kwargs,
            device=self.device,
        ).policy
        self.global_version = 0
        self.lock = threading.Lock()
        self.stop_event = threading.Event()

    def serve_loop(self):
        env = build_env(self.client_id, self.params, training=False, role="serve-loop")
        obs, _ = env.reset()
        while not self.stop_event.is_set():
            with self.lock:
                actions, _ = self.policy_serve.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(actions)
            if dones.any():
                obs, _ = env.reset()

    def eval_policy(self, policy) -> float:
        env = build_env(self.client_id, self.params, training=False, role="eval")
        obs, _ = env.reset()
        total_rew = 0.0
        steps = 0
        while steps < self.eval_steps:
            actions, _ = policy.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(actions)
            total_rew += float(rewards.mean())
            steps += 1
            if dones.any():
                obs, _ = env.reset()
        env.close()
        return total_rew / max(steps, 1)

    def send_update(self, update: Dict):
        state = update["state_dict"]
        weight = float(max(update.get("weight", 0.0), 1))
        meta = {
            "client_id": self.client_id,
            "weight": weight,
            "metrics": update.get("metrics", {}),
        }
        fname = f"{self.client_id}_{int(time.time())}"
        meta_path = self.updates_dir / f"{fname}.json"
        state_path = self.updates_dir / f"{fname}.pt"
        torch.save(state, state_path)
        meta_path.write_text(json.dumps(meta, indent=2))

    def maybe_pull_global(self):
        latest = self.ckpt_dir / "latest_global.json"
        if not latest.exists():
            return None, None
        meta = json.loads(latest.read_text())
        version = int(meta.get("version", 0))
        if version <= self.global_version:
            return None, None
        state_path = Path(meta["path"])
        if not state_path.exists():
            return None, None
        return version, torch.load(state_path, map_location="cpu")

    def training_loop(self):
        raise RuntimeError(REALTIME_DISABLED_MSG)

        while not self.stop_event.is_set():
            # train chunk
            update = self.client.train_local(self.chunk_steps, callback=None).__dict__
            self.send_update(update)

            # pull global
            version, global_state = self.maybe_pull_global()
            if version is not None and global_state is not None:
                with self.lock:
                    # soft merge
                    current = self.client.get_policy_state()
                    merged = {k: (1 - self.alpha_merge) * current[k] + self.alpha_merge * global_state[k] for k in current}
                    self.client.load_policy_state(merged)
                    self.global_version = version

            # gating/promotion
            with self.lock:
                eval_new = self.eval_policy(self.client.model.policy)
                eval_old = self.eval_policy(self.policy_serve)
                if eval_new >= eval_old - self.gating_delta:
                    self.policy_serve.load_state_dict(self.client.model.policy.state_dict())
                    torch.save(self.policy_serve.state_dict(), self.ckpt_dir / f"{self.client_id}_best.pt")
                else:
                    pass  # keep serving policy

    def start(self):
        raise RuntimeError(REALTIME_DISABLED_MSG)

        self.updates_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        t1 = threading.Thread(target=self.serve_loop, daemon=True)
        t1.start()
        self.training_loop()


def main():
    ap = argparse.ArgumentParser(description="Realtime FL PPO client (file-based)")
    ap.add_argument("--client-id", required=True)
    ap.add_argument("--env-name", type=str, default="4nodes")
    ap.add_argument("--tls-action-type", type=str, default="next_or_not")
    ap.add_argument("--tls-id", type=str, default=None, help="If provided, overrides client-id for env tls_id")
    ap.add_argument("--num-seconds", type=int, default=500)
    ap.add_argument("--chunk-steps", type=int, default=2000)
    ap.add_argument("--alpha-merge", type=float, default=0.1)
    ap.add_argument("--gating-delta", type=float, default=0.0)
    ap.add_argument("--eval-steps", type=int, default=200)
    ap.add_argument("--updates-dir", type=str, default="federated/updates")
    ap.add_argument("--checkpoints-dir", type=str, default="federated/checkpoints")
    ap.add_argument("--use-gui", action="store_true")
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    raise RuntimeError(REALTIME_DISABLED_MSG)

    env_name = args.env_name
    tls_id = args.tls_id or args.client_id
    sumo_cfg = f"Scenario/{env_name}/env/vehicle.sumocfg"
    net_file = f"Scenario/{env_name}/env/{env_name}.net.xml"
    phase_num = 2 if args.tls_action_type == "next_or_not" else 4

    params = {
        "tls_id": tls_id,
        "num_seconds": args.num_seconds,
        "tls_action_type": args.tls_action_type,
        "phase_num": phase_num,
        "sumo_cfg": sumo_cfg,
        "net_file": net_file,
        "use_gui": args.use_gui,
        "log_file": f"federated/logs/{tls_id}",
        "label_prefix": f"rt-{args.client_id}",
    }

    client = RealtimeClient(
        client_id=args.client_id,
        params=params,
        updates_dir=Path(args.updates_dir),
        checkpoints_dir=Path(args.checkpoints_dir),
        chunk_steps=args.chunk_steps,
        alpha_merge=args.alpha_merge,
        gating_delta=args.gating_delta,
        eval_steps=args.eval_steps,
        device=args.device,
    )
    client.start()


if __name__ == "__main__":
    main()
