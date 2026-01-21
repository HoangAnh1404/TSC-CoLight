#!/usr/bin/env python3
"""
Extract a k-hop subnet around a TLS and (optionally) filter routes for it.

Requirements:
- sumolib and netconvert in PATH (from SUMO).

Usage example:
python Scenario/extract_subnet.py \
  --net Scenario/4nodes/env/4nodes.net.xml \
  --tls-id J2 --hop 1 \
  --out-dir Scenario/J2/env \
  --routes Scenario/4nodes/routes/vehicle.rou.xml \
  --routes-out Scenario/J2/routes/vehicle.rou.xml
"""
from __future__ import annotations

import argparse
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional, Set, Tuple

import sumolib


def bfs_nodes(net: sumolib.net.Net, start_tls: str, hop: int) -> Set[str]:
    """Return node ids within hop distance (inclusive) from start TLS."""
    if start_tls not in {n.getID() for n in net.getTrafficLights()}:
        raise ValueError(f"TLS id '{start_tls}' not found in net.")
    frontier = {start_tls}
    visited = {start_tls}
    for _ in range(max(hop, 0)):
        nxt: Set[str] = set()
        for nid in frontier:
            node = net.getNode(nid)
            # neighbors via incoming/outgoing edges
            for e in list(node.getIncoming()) + list(node.getOutgoing()):
                n_from = e.getFromNode().getID()
                n_to = e.getToNode().getID()
                if n_from not in visited:
                    nxt.add(n_from)
                if n_to not in visited:
                    nxt.add(n_to)
        frontier = nxt
        visited |= nxt
        if not frontier:
            break
    return visited


def edges_within_nodes(net: sumolib.net.Net, keep_nodes: Set[str]) -> List[str]:
    ids: List[str] = []
    for e in net.getEdges(withInternal=False):
        if e.getFromNode().getID() in keep_nodes and e.getToNode().getID() in keep_nodes:
            ids.append(e.getID())
    return ids


def run_netconvert(src_net: Path, keep_edges: List[str], dst_net: Path) -> None:
    keep_file = dst_net.with_suffix(".keep.txt")
    keep_file.write_text("\n".join(keep_edges))
    cmd = [
        "netconvert",
        "--sumo-net-file",
        str(src_net),
        "--keep-edges.input-file",
        str(keep_file),
        "--output-file",
        str(dst_net),
        "--remove-edges.isolated",
    ]
    subprocess.run(cmd, check=True)


# ---------------- Route filtering (self contained) ---------------- #
def read_net_edges(net_path: str) -> Set[str]:
    tree = ET.parse(net_path)
    root = tree.getroot()
    edges: Set[str] = set()
    for e in root.findall("edge"):
        eid = e.get("id")
        if eid:
            edges.add(eid)
    return edges


def longest_contiguous_subseq_in_set(seq: List[str], allowed: Set[str]) -> Optional[Tuple[int, int]]:
    best = None
    best_len = 0
    cur_start = None
    for i, e in enumerate(seq):
        if e in allowed:
            if cur_start is None:
                cur_start = i
        else:
            if cur_start is not None:
                cur_len = i - cur_start
                if cur_len > best_len:
                    best_len = cur_len
                    best = (cur_start, i - 1)
                cur_start = None
    if cur_start is not None:
        cur_len = len(seq) - cur_start
        if cur_len > best_len:
            best_len = cur_len
            best = (cur_start, len(seq) - 1)
    return best


def pretty_write(tree: ET.ElementTree, path: str) -> None:
    try:
        ET.indent(tree, space="    ", level=0)  # type: ignore[attr-defined]
    except Exception:
        pass
    tree.write(path, encoding="UTF-8", xml_declaration=True)


def filter_routes_for_net(
    net_path: str,
    routes_path: str,
    out_rou: str,
    out_trips: Optional[str] = None,
    min_seg_len: int = 2,
) -> None:
    allowed_edges = read_net_edges(net_path)
    src_tree = ET.parse(routes_path)
    src_root = src_tree.getroot()

    out_routes_root = ET.Element("routes")
    out_routes_root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    out_routes_root.set("xsi:noNamespaceSchemaLocation", "http://sumo.dlr.de/xsd/routes_file.xsd")

    out_trips_root = None
    if out_trips:
        out_trips_root = ET.Element("routes")
        out_trips_root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
        out_trips_root.set("xsi:noNamespaceSchemaLocation", "http://sumo.dlr.de/xsd/routes_file.xsd")

    # copy vType definitions
    for child in list(src_root):
        if child.tag == "vType":
            out_routes_root.append(child)
            if out_trips_root is not None:
                out_trips_root.append(child)

    total = kept_full = projected = skipped = 0
    for veh in src_root.findall("vehicle"):
        total += 1
        route_el = veh.find("route")
        if route_el is None:
            skipped += 1
            continue
        edges_str = route_el.get("edges")
        if not edges_str:
            skipped += 1
            continue
        edges = edges_str.split()

        if all(e in allowed_edges for e in edges):
            out_routes_root.append(veh)
            kept_full += 1
            continue

        if out_trips_root is not None:
            seg = longest_contiguous_subseq_in_set(edges, allowed_edges)
            if seg is None:
                skipped += 1
                continue
            s, t = seg
            seg_edges = edges[s : t + 1]
            if len(seg_edges) < min_seg_len:
                skipped += 1
                continue
            trip = ET.Element("trip")
            trip.set("id", veh.get("id", f"veh_{total}"))
            if veh.get("type"):
                trip.set("type", veh.get("type"))
            if veh.get("depart"):
                trip.set("depart", veh.get("depart"))
            if veh.get("departLane"):
                trip.set("departLane", veh.get("departLane"))
            trip.set("from", seg_edges[0])
            trip.set("to", seg_edges[-1])
            out_trips_root.append(trip)
            projected += 1
        else:
            skipped += 1

    pretty_write(ET.ElementTree(out_routes_root), out_rou)
    if out_trips_root is not None and out_trips:
        pretty_write(ET.ElementTree(out_trips_root), out_trips)

    print(
        f"Routes filtered: total={total}, kept_full={kept_full}, "
        f"projected={projected if out_trips else 0}, skipped={skipped}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True, help="Source .net.xml")
    ap.add_argument("--tls-id", required=True, help="Target TLS id")
    ap.add_argument("--hop", type=int, default=1, help="k-hop neighborhood to keep")
    ap.add_argument("--out-dir", required=True, help="Output directory for subnet net.xml")
    ap.add_argument("--routes", default=None, help="Optional source routes .rou.xml")
    ap.add_argument("--routes-out", default=None, help="Optional output routes .rou.xml (filtered)")
    ap.add_argument("--trips-out", default=None, help="Optional output trips .trips.xml for partial routes")
    args = ap.parse_args()

    src_net = Path(args.net).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    dst_net = out_dir / f"{args.tls_id}.net.xml"

    net = sumolib.net.readNet(str(src_net))
    keep_nodes = bfs_nodes(net, args.tls_id, args.hop)
    keep_edges = edges_within_nodes(net, keep_nodes)
    if not keep_edges:
        raise RuntimeError("No edges selected; check tls-id or hop.")

    run_netconvert(src_net, keep_edges, dst_net)
    print(f"Subnet net saved to {dst_net} (edges kept: {len(keep_edges)})")

    if args.routes and args.routes_out:
        filter_routes_for_net(
            net_path=str(dst_net),
            routes_path=str(Path(args.routes).resolve()),
            out_rou=str(Path(args.routes_out).resolve()),
            out_trips=str(Path(args.trips_out).resolve()) if args.trips_out else None,
        )


if __name__ == "__main__":
    main()
