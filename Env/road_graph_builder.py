from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

try:
    import sumolib
except ImportError:  # pragma: no cover - optional dependency
    sumolib = None

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None


@dataclass
class GraphTopology:
    """Pure topology: ids, adjacency, edge_index; no neighborhood filtering."""

    tls_id_to_idx: Dict[str, int]
    idx_to_tls_id: List[str]
    edge_index: np.ndarray  # shape (2, E) compatible with PyG
    adj_matrix: np.ndarray  # shape (N, N) dense adjacency with 0/1 entries
    tls_xy: Optional[List[Optional[Tuple[float, float]]]] = None
    node_id_list: Optional[List[str]] = None

    @property
    def num_nodes(self) -> int:
        return len(self.idx_to_tls_id)

    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1]

    def to_graph_spec(self, neighbors_list: List[List[int]]) -> "GraphSpec":
        return GraphSpec(
            tls_id_to_idx=self.tls_id_to_idx,
            idx_to_tls_id=self.idx_to_tls_id,
            edge_index=self.edge_index,
            adj_matrix=self.adj_matrix,
            neighbors_list=neighbors_list,
            tls_xy=self.tls_xy,
            node_id_list=self.node_id_list or self.idx_to_tls_id,
        )


@dataclass
class GraphSpec:
    """Container describing the fixed road graph used during training."""

    tls_id_to_idx: Dict[str, int]
    idx_to_tls_id: List[str]
    edge_index: np.ndarray  # shape (2, E) compatible with PyG
    adj_matrix: np.ndarray  # shape (N, N) dense adjacency with 0/1 entries
    neighbors_list: List[List[int]]
    tls_xy: Optional[List[Optional[Tuple[float, float]]]] = None
    node_id_list: Optional[List[str]] = None

    def edge_index_as_tensor(self):
        """Return `edge_index` as a torch.LongTensor, if torch is installed."""
        if torch is None:
            raise ImportError("PyTorch is required to convert `edge_index` to a tensor.")
        return torch.as_tensor(self.edge_index, dtype=torch.long)

    def neighbors_of(self, tls: Any) -> List[int]:
        """Return neighbor indices given a tls id or index."""
        idx = tls if isinstance(tls, int) else self.tls_id_to_idx[tls]
        return self.neighbors_list[idx]

    def neighbors_edge_index(self, exclude_self: bool = False, add_reverse: bool = False, add_self_loops: bool = False):
        """
        Build edge_index from neighbors_list: edge i -> neighbor_j.

        Args:
            exclude_self: Drop self-edges if present in neighbors_list.
            add_reverse: Add reverse edges for bidirectional exchange.
            add_self_loops: Ensure self-loops are present.
        """
        edges = set()
        for i, nbrs in enumerate(self.neighbors_list):
            for j in nbrs:
                if exclude_self and i == j:
                    continue
                edges.add((i, j))
                if add_reverse and i != j:
                    edges.add((j, i))
        if add_self_loops:
            for i in range(len(self.neighbors_list)):
                edges.add((i, i))

        if edges:
            rows, cols = zip(*sorted(edges))
            edge_arr = np.vstack((rows, cols)).astype(np.int64)
        else:
            edge_arr = np.zeros((2, 0), dtype=np.int64)
        if torch is None:
            return edge_arr
        return torch.as_tensor(edge_arr, dtype=torch.long)

    def as_topology(self) -> GraphTopology:
        return GraphTopology(
            tls_id_to_idx=self.tls_id_to_idx,
            idx_to_tls_id=self.idx_to_tls_id,
            edge_index=self.edge_index,
            adj_matrix=self.adj_matrix,
            tls_xy=self.tls_xy,
            node_id_list=self.node_id_list or self.idx_to_tls_id,
        )

    @property
    def num_nodes(self) -> int:
        return len(self.idx_to_tls_id)

    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1]


class RoadGraphBuilder:
    """
    Build a static road graph from SUMO networks or pre-defined edges.

    The output `GraphSpec` contains deterministic mappings and adjacency
    representations (edge_index, adj_matrix, neighbors_list) that can be reused
    across the whole training run.
    """

    # ---- Public API ----------------------------------------------------- #
    @staticmethod
    def build_from_net_file(
        net_file: str,
        tls_ids: Optional[List[str]] = None,
        directed: bool = True,
        make_bidirectional: bool = False,
        include_self_loops: bool = False,
        max_hops_between_tls: int = 1,
        neighbor_strategy: str = "hop",
        neighbor_hop_k: int = 1,
        neighbor_top_k: Optional[int] = None,
        include_self_in_neighbor: bool = True,
    ) -> GraphSpec:
        """
        Build graph spec directly from a SUMO .net.xml file.

        Args:
            net_file: Path to SUMO net file.
            tls_ids: Optional explicit ordering of tls ids. If None, all tls in
                the net are used in sorted order.
            directed: Keep the graph directed (recommended). Set to False to
                symmetrize edges.
            make_bidirectional: Add reverse edges regardless of `directed`
                flag; useful if the net is missing reverse edges.
            include_self_loops: Add self loops to adjacency/edge_index.
            max_hops_between_tls: How many hops through non-tls nodes are
                allowed when searching for neighbors. Set to 1 to only connect
                tls pairs sharing a direct edge.
            neighbor_strategy: `hop` (BFS hop distance) or `distance`
                (geometric, requires TLS coordinates); falls back to hop if
                distance is unavailable.
            neighbor_hop_k: Maximum hop distance to consider when building
                neighborhoods (only applies to neighbor selection).
            neighbor_top_k: Cap total neighbors per node (the returned list),
                counting self if `include_self_in_neighbor` is True.
            include_self_in_neighbor: Always include self in neighborhood.
        """
        topology = RoadGraphBuilder.build_topology_from_net_file(
            net_file=net_file,
            tls_ids=tls_ids,
            directed=directed,
            make_bidirectional=make_bidirectional,
            include_self_loops=include_self_loops,
            max_hops_between_tls=max_hops_between_tls,
        )
        neighbors = NeighborhoodSelector.build_neighbors(
            topology=topology,
            strategy=neighbor_strategy,
            hop_k=neighbor_hop_k,
            top_k=neighbor_top_k,
            include_self=include_self_in_neighbor,
        )
        return topology.to_graph_spec(neighbors)

    @staticmethod
    def build_from_sumo_net(
        net: Any,
        tls_ids: Optional[List[str]] = None,
        directed: bool = True,
        make_bidirectional: bool = False,
        include_self_loops: bool = False,
        max_hops_between_tls: int = 1,
        neighbor_strategy: str = "hop",
        neighbor_hop_k: int = 1,
        neighbor_top_k: Optional[int] = None,
        include_self_in_neighbor: bool = True,
    ) -> GraphSpec:
        """
        Build graph spec from an in-memory `sumolib.net.Net` object.

        Args mirror `build_from_net_file`.
        """
        topology = RoadGraphBuilder.build_topology_from_sumo_net(
            net=net,
            tls_ids=tls_ids,
            directed=directed,
            make_bidirectional=make_bidirectional,
            include_self_loops=include_self_loops,
            max_hops_between_tls=max_hops_between_tls,
        )
        neighbors = NeighborhoodSelector.build_neighbors(
            topology=topology,
            strategy=neighbor_strategy,
            hop_k=neighbor_hop_k,
            top_k=neighbor_top_k,
            include_self=include_self_in_neighbor,
        )
        return topology.to_graph_spec(neighbors)

    @staticmethod
    def build_from_edges(
        edges: Iterable[Tuple[str, str]],
        tls_ids: Optional[Iterable[str]] = None,
        directed: bool = True,
        make_bidirectional: bool = False,
        include_self_loops: bool = False,
        neighbor_strategy: str = "hop",
        neighbor_hop_k: int = 1,
        neighbor_top_k: Optional[int] = None,
        include_self_in_neighbor: bool = True,
    ) -> GraphSpec:
        """
        Build graph spec from user-provided edges (src_tls_id, dst_tls_id).

        This is useful when the topology is already available (e.g., exported
        from Tshub) and avoids re-reading the SUMO network.
        """
        topology = RoadGraphBuilder.build_topology_from_edges(
            edges=edges,
            tls_ids=tls_ids,
            directed=directed,
            make_bidirectional=make_bidirectional,
            include_self_loops=include_self_loops,
        )
        neighbors = NeighborhoodSelector.build_neighbors(
            topology=topology,
            strategy=neighbor_strategy,
            hop_k=neighbor_hop_k,
            top_k=neighbor_top_k,
            include_self=include_self_in_neighbor,
        )
        return topology.to_graph_spec(neighbors)

    # ---- Topology builders (no neighborhood selection) ------------------ #
    @staticmethod
    def build_topology_from_net_file(
        net_file: str,
        tls_ids: Optional[List[str]] = None,
        directed: bool = True,
        make_bidirectional: bool = False,
        include_self_loops: bool = False,
        max_hops_between_tls: int = 1,
    ) -> GraphTopology:
        if sumolib is None:
            raise ImportError(
                "sumolib is required to parse SUMO net files. Please install "
                "SUMO (or add $SUMO_HOME/tools to PYTHONPATH) to use this helper."
            )
        net = sumolib.net.readNet(net_file)
        return RoadGraphBuilder.build_topology_from_sumo_net(
            net=net,
            tls_ids=tls_ids,
            directed=directed,
            make_bidirectional=make_bidirectional,
            include_self_loops=include_self_loops,
            max_hops_between_tls=max_hops_between_tls,
        )

    @staticmethod
    def build_topology_from_sumo_net(
        net: Any,
        tls_ids: Optional[List[str]] = None,
        directed: bool = True,
        make_bidirectional: bool = False,
        include_self_loops: bool = False,
        max_hops_between_tls: int = 1,
    ) -> GraphTopology:
        tls_id_list = RoadGraphBuilder._resolve_tls_ids(net, tls_ids)
        tls_xy = RoadGraphBuilder._resolve_tls_coords(net, tls_id_list)
        adjacency = RoadGraphBuilder._extract_node_adjacency(net)
        tls_edges = RoadGraphBuilder._collapse_edges_to_tls(
            adjacency=adjacency,
            tls_ids=tls_id_list,
            max_hops=max_hops_between_tls,
        )
        return RoadGraphBuilder.build_topology_from_edges(
            edges=tls_edges,
            tls_ids=tls_id_list,
            directed=directed,
            make_bidirectional=make_bidirectional,
            include_self_loops=include_self_loops,
            tls_xy=tls_xy,
        )

    @staticmethod
    def build_topology_from_edges(
        edges: Iterable[Tuple[str, str]],
        tls_ids: Optional[Iterable[str]] = None,
        directed: bool = True,
        make_bidirectional: bool = False,
        include_self_loops: bool = False,
        tls_xy: Optional[List[Optional[Tuple[float, float]]]] = None,
    ) -> GraphTopology:
        edge_pairs: Set[Tuple[str, str]] = {(str(src), str(dst)) for src, dst in edges}
        if tls_ids is None:
            tls_id_list = sorted({node for pair in edge_pairs for node in pair})
        else:
            tls_id_list = list(tls_ids)
        tls_set = set(tls_id_list)

        unknown_nodes = {node for pair in edge_pairs for node in pair if node not in tls_set}
        if unknown_nodes:
            raise ValueError(
                f"Edges reference tls ids not present in provided tls_ids: {sorted(unknown_nodes)}"
            )

        tls_id_to_idx = {tls_id: idx for idx, tls_id in enumerate(tls_id_list)}
        num_nodes = len(tls_id_list)
        adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.int64)

        def add_edge(src_id: str, dst_id: str):
            src_idx = tls_id_to_idx[src_id]
            dst_idx = tls_id_to_idx[dst_id]
            if not include_self_loops and src_idx == dst_idx:
                return
            adj_matrix[src_idx, dst_idx] = 1

        for src_id, dst_id in edge_pairs:
            add_edge(src_id, dst_id)
            if not directed or make_bidirectional:
                add_edge(dst_id, src_id)

        if include_self_loops:
            for tls_id in tls_id_list:
                add_edge(tls_id, tls_id)

        rows, cols = np.nonzero(adj_matrix)
        edge_index = np.vstack((rows, cols)).astype(np.int64) if rows.size else np.zeros((2, 0), dtype=np.int64)

        if tls_xy is not None and len(tls_xy) != num_nodes:
            raise ValueError("Length of tls_xy must match number of tls nodes.")

        return GraphTopology(
            tls_id_to_idx=tls_id_to_idx,
            idx_to_tls_id=tls_id_list,
            edge_index=edge_index,
            adj_matrix=adj_matrix,
            tls_xy=tls_xy,
            node_id_list=tls_id_list,
        )

    # ---- Helpers -------------------------------------------------------- #
    @staticmethod
    def _resolve_tls_ids(net: Any, tls_ids: Optional[List[str]]) -> List[str]:
        available_tls_ids = sorted({tls.getID() for tls in net.getTrafficLights()})
        if tls_ids is None:
            return available_tls_ids

        missing = [tls_id for tls_id in tls_ids if tls_id not in available_tls_ids]
        if missing:
            raise ValueError(f"TLS ids not found in network: {missing}")
        return list(tls_ids)

    @staticmethod
    def _resolve_tls_coords(net: Any, tls_ids: List[str]) -> List[Optional[Tuple[float, float]]]:
        coords: List[Optional[Tuple[float, float]]] = []
        for tls_id in tls_ids:
            node = net.getNode(tls_id) if hasattr(net, "getNode") else None
            if node is None:
                coords.append(None)
            else:
                try:
                    coords.append(node.getCoord())
                except Exception:
                    coords.append(None)
        return coords

    @staticmethod
    def _extract_node_adjacency(net: Any) -> Dict[str, Set[str]]:
        """Return adjacency between all nodes (including non-tls) from the net."""
        adjacency: Dict[str, Set[str]] = {}
        for edge in net.getEdges():
            try:
                if edge.getFunction() == "internal":
                    continue
            except Exception:
                pass

            from_node = edge.getFromNode().getID()
            to_node = edge.getToNode().getID()
            adjacency.setdefault(from_node, set()).add(to_node)
        return adjacency

    @staticmethod
    def _collapse_edges_to_tls(
        adjacency: Dict[str, Set[str]],
        tls_ids: List[str],
        max_hops: int,
    ) -> Set[Tuple[str, str]]:
        """
        Walk the node graph and connect tls nodes within `max_hops` steps.

        Non-tls nodes are treated as intermediate waypoints and are not part of
        the final graph.
        """
        tls_set = set(tls_ids)
        max_hops = max(1, int(max_hops))
        tls_edges: Set[Tuple[str, str]] = set()

        for start_tls in tls_set:
            visited: Set[str] = set()
            frontier = deque([(nbr, 1) for nbr in adjacency.get(start_tls, set())])
            while frontier:
                node_id, depth = frontier.popleft()
                if node_id in visited:
                    continue
                visited.add(node_id)

                if node_id in tls_set:
                    tls_edges.add((start_tls, node_id))
                    continue

                if depth >= max_hops:
                    continue

                for next_node in adjacency.get(node_id, set()):
                    frontier.append((next_node, depth + 1))

        return tls_edges


class NeighborhoodSelector:
    """
    Select neighborhoods on top of a fixed graph topology.

    Separating this from topology building makes it easy to try different
    neighbor pruning policies (top-k, hop cutoff, geometric distance).
    """

    @staticmethod
    def build_neighbors(
        topology: GraphTopology,
        strategy: str = "hop",
        hop_k: int = 1,
        top_k: Optional[int] = None,
        include_self: bool = True,
    ) -> List[List[int]]:
        """
        Select neighbors for each node based on the provided strategy.

        Args:
            strategy: 'hop' (BFS distance) or 'distance' (euclidean, falls back
                to hop if coordinates are missing).
            hop_k: Maximum hop distance considered.
            top_k: Cap the returned neighbor list length. The cap counts self
                when `include_self` is True.
            include_self: If True, the node itself is always included first.
        """
        strategy = strategy.lower()
        if strategy not in {"hop", "distance"}:
            raise ValueError(f"Unknown neighbor strategy: {strategy}")

        adj_list = [set(np.nonzero(topology.adj_matrix[row_idx])[0].tolist()) for row_idx in range(topology.num_nodes)]

        neighbors: List[List[int]] = []
        for start_idx in range(topology.num_nodes):
            hop_dist = NeighborhoodSelector._bfs_hops(adj_list, start_idx)

            if strategy == "distance" and NeighborhoodSelector._coords_available(topology.tls_xy):
                ordered_nodes = NeighborhoodSelector._order_by_distance(
                    start_idx=start_idx,
                    hop_dist=hop_dist,
                    tls_xy=topology.tls_xy,  # type: ignore[arg-type]
                    hop_k=hop_k,
                )
                if not ordered_nodes:
                    ordered_nodes = NeighborhoodSelector._order_by_hop(hop_dist, hop_k)
            else:
                ordered_nodes = NeighborhoodSelector._order_by_hop(hop_dist, hop_k)

            node_list: List[int] = []
            if include_self:
                node_list.append(start_idx)

            for node in ordered_nodes:
                if node == start_idx:
                    continue
                node_list.append(node)
                if top_k is not None and len(node_list) >= top_k:
                    break

            neighbors.append(node_list)
        return neighbors

    @staticmethod
    def _bfs_hops(adj_list: List[Set[int]], start_idx: int) -> Dict[int, int]:
        dist: Dict[int, int] = {start_idx: 0}
        queue: deque = deque([start_idx])
        while queue:
            current = queue.popleft()
            for nxt in adj_list[current]:
                if nxt not in dist:
                    dist[nxt] = dist[current] + 1
                    queue.append(nxt)
        return dist

    @staticmethod
    def _order_by_hop(hop_dist: Dict[int, int], hop_k: int) -> List[int]:
        filtered = [(node, dist) for node, dist in hop_dist.items() if dist <= hop_k]
        filtered.sort(key=lambda x: (x[1], x[0]))
        return [node for node, _ in filtered]

    @staticmethod
    def _coords_available(tls_xy: Optional[List[Optional[Tuple[float, float]]]]) -> bool:
        return (
            tls_xy is not None
            and any(coord is not None for coord in tls_xy)
            and all(coord is None or (len(coord) == 2) for coord in tls_xy)
        )

    @staticmethod
    def _order_by_distance(
        start_idx: int,
        hop_dist: Dict[int, int],
        tls_xy: List[Optional[Tuple[float, float]]],
        hop_k: int,
    ) -> List[int]:
        if tls_xy[start_idx] is None:
            return []

        x0, y0 = tls_xy[start_idx]
        candidates = []
        for node, dist_hop in hop_dist.items():
            if dist_hop > hop_k:
                continue
            coord = tls_xy[node]
            if coord is None:
                continue
            dx = x0 - coord[0]
            dy = y0 - coord[1]
            candidates.append((math.hypot(dx, dy), dist_hop, node))

        candidates.sort(key=lambda x: (x[0], x[1], x[2]))
        return [node for _, _, node in candidates]
