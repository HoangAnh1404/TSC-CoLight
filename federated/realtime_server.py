from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

# Ensure project root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from FL_PPO.fed_ppo_fedavg import FederatedPPOCoordinator, ClientUpdate

REALTIME_DISABLED_MSG = "Realtime training has been disabled; use run_federated.py for standard PPO FedAvg."


def load_update(meta_path: Path) -> Optional[ClientUpdate]:
    try:
        meta = json.loads(meta_path.read_text())
        state_path = meta_path.with_suffix(".pt")
        if not state_path.exists():
            return None
        state_dict = torch.load(state_path, map_location="cpu")
        return ClientUpdate(
            tls_id=meta["client_id"],
            state_dict=state_dict,
            weight=float(meta.get("weight", 1.0)),
            metrics=meta.get("metrics", {}),
        )
    except Exception as exc:
        print(f"[WARN] Failed to load update {meta_path}: {exc}")
        return None


def save_global(global_state: Dict[str, torch.Tensor], out_dir: Path, version: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(global_state, out_dir / f"global_v{version}.pt")
    latest = {
        "version": version,
        "path": str(out_dir / f"global_v{version}.pt"),
        "timestamp": time.time(),
    }
    (out_dir / "latest_global.json").write_text(json.dumps(latest, indent=2))


def main():
    raise RuntimeError(REALTIME_DISABLED_MSG)

    ap = argparse.ArgumentParser(description="Realtime FedAvg server (file-based)")
    ap.add_argument("--updates-dir", type=str, default="federated/updates", help="Directory where clients write updates")
    ap.add_argument("--checkpoints-dir", type=str, default="federated/checkpoints", help="Directory to save global checkpoints")
    ap.add_argument("--poll-interval", type=float, default=5.0, help="Seconds between polls")
    ap.add_argument("--aggregate-every", type=int, default=1, help="Aggregate after this many new updates (>=1)")
    ap.add_argument("--alpha", type=float, default=0.0, help="Mixing with previous global")
    args = ap.parse_args()

    updates_dir = Path(args.updates_dir)
    updates_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.checkpoints_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    coord = FederatedPPOCoordinator(alpha=args.alpha)
    global_state: Optional[Dict[str, torch.Tensor]] = None
    global_version = 0

    seen: set[Path] = set()
    pending: List[ClientUpdate] = []

    print("[INFO] Realtime server started.")
    while True:
        # poll new updates
        for meta_path in updates_dir.glob("*.json"):
            if meta_path in seen:
                continue
            seen.add(meta_path)
            upd = load_update(meta_path)
            if upd is not None:
                pending.append(upd)
        if pending and len(pending) >= args.aggregate_every:
            global_state = coord.aggregate(pending, prev_global=global_state)
            global_version += 1
            save_global(global_state, ckpt_dir, global_version)
            print(f"[AGG] version={global_version} aggregated {len(pending)} updates")
            pending.clear()
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
