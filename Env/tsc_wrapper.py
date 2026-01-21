'''
@Author: WANG Maonan
@Date: 2023-10-30 14:22:45
@Description: 计算 tsc env 中的 state 和 reward
@LastEditTime: 2024-04-24 22:06:50
'''
import time
import numpy as np
import gymnasium as gym
from itertools import chain
from loguru import logger
from gymnasium.core import Env
from collections import deque, defaultdict
from typing import Any, SupportsFloat, Tuple, Dict, List, Optional
from stable_baselines3.common.monitor import ResultsWriter


class VehicleIDList:
    def __init__(self) -> None:
        self.elements = []

    def add_element(self, element) -> None:
        """合并不同时刻的 vehicle_ids
        -> a = [[1,2], [3,4]]
        -> b = [[5,6], [7,8]]
        -> a+b = [[1, 2], [3, 4], [5, 6], [7, 8]]
        """
        self.elements += element

    def clear_elements(self) -> None:
        self.elements = []

    def flatten_remove_duplicates_elements(self):
        flattened_list = list(chain(*self.elements))  # 展开列表
        unique_list = list(set(flattened_list))  # 去除重复的元素
        self.clear_elements()
        return unique_list


class Occupancy:
    """Compute lane occupancy using actual vehicle dimensions."""

    def __init__(self) -> None:
        self.lane_area_cache: Dict[str, float] = {}

    @staticmethod
    def _polygon_area(points) -> float:
        """Compute polygon area with the shoelace formula."""
        if not points or len(points) < 3:
            return 0.0
        area = 0.0
        num_points = len(points)
        for i in range(num_points):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % num_points]
            area += x1 * y2 - x2 * y1
        return abs(area) / 2.0

    def _get_lane_area(self, lane_id: str, lane_state: Dict[str, Dict[str, Any]]) -> float:
        """Return cached lane area or compute it from the lane polygon."""
        if lane_id in self.lane_area_cache:
            return self.lane_area_cache[lane_id]

        lane_info = lane_state.get(lane_id, {})
        shape = lane_info.get('shape')
        if not shape:
            return 0.0

        area = self._polygon_area(shape)
        self.lane_area_cache[lane_id] = area
        return area

    def calculate(self, tls_state: Dict[str, Any], vehicle_state: Dict[str, Dict[str, Any]], lane_state: Dict[str, Dict[str, Any]]) -> List[float]:
        """Calculate occupancy (percentage) for each movement using vehicle size and lane area."""
        vehicle_state = vehicle_state or {}
        lane_state = lane_state or {}
        movement_ids = tls_state.get('movement_ids') or []
        if not movement_ids:
            return tls_state.get('last_step_occupancy', [])

        movement_lane_ids = tls_state.get('movement_lane_ids') or {}

        vehicles_by_lane = defaultdict(list)
        for veh_info in vehicle_state.values():
            lane_id = veh_info.get('lane_id')
            if lane_id is not None:
                vehicles_by_lane[lane_id].append(veh_info)

        occupancies = []
        for movement_id in movement_ids:
            lane_ids = movement_lane_ids.get(movement_id, [])
            lane_area = 0.0
            vehicle_area = 0.0

            for lane_id in lane_ids:
                _lane_area = self._get_lane_area(lane_id, lane_state)
                if _lane_area <= 0:
                    continue
                lane_area += _lane_area
                for veh in vehicles_by_lane.get(lane_id, []):
                    length = veh.get('length') or 0.0
                    width = veh.get('width') or 0.0
                    if (length <= 0) or (width <= 0):
                        continue
                    vehicle_area += length * width

            occupancy_ratio = vehicle_area / lane_area if lane_area > 0 else 0.0
            occupancy_percent = min(max(occupancy_ratio * 100, 0.0), 100.0)
            occupancies.append(float(occupancy_percent))

        return occupancies

    def calculate_edges(
        self,
        incoming_edges: List[str],
        edge_to_lanes: Dict[str, List[str]],
        vehicle_state: Optional[Dict[str, Dict[str, Any]]],
        lane_state: Optional[Dict[str, Dict[str, Any]]],
    ) -> List[float]:
        """Calculate occupancy (percentage) per incoming edge using lane areas."""
        vehicle_state = vehicle_state or {}
        lane_state = lane_state or {}

        vehicles_by_lane = defaultdict(list)
        for veh_info in vehicle_state.values():
            lane_id = veh_info.get("lane_id")
            if lane_id is not None:
                vehicles_by_lane[lane_id].append(veh_info)

        occupancies: List[float] = []
        for edge_id in incoming_edges:
            lane_ids = edge_to_lanes.get(edge_id, [])
            lane_area = 0.0
            vehicle_area = 0.0

            for lane_id in lane_ids:
                _lane_area = self._get_lane_area(lane_id, lane_state)
                if _lane_area <= 0:
                    continue
                lane_area += _lane_area
                for veh in vehicles_by_lane.get(lane_id, []):
                    length = veh.get("length") or 0.0
                    width = veh.get("width") or 0.0
                    if (length <= 0) or (width <= 0):
                        continue
                    vehicle_area += length * width

            occupancy_ratio = vehicle_area / lane_area if lane_area > 0 else 0.0
            occupancy_percent = min(max(occupancy_ratio * 100, 0.0), 100.0)
            occupancies.append(float(occupancy_percent))

        return occupancies


class OccupancyList:
    def __init__(self) -> None:
        self.elements = []

    def add_element(self, element) -> None:
        if isinstance(element, list):
            cleaned = [float(e) for e in element]
            self.elements.append(cleaned)
        else:
            raise TypeError("添加的元素必须是列表类型")

    def clear_elements(self) -> None:
        self.elements = []

    def calculate_average(self) -> float:
        """计算一段时间的平均 occupancy
        """
        arr = np.array(self.elements)
        averages = np.mean(arr, axis=0, dtype=np.float32)/100
        self.clear_elements() # 清空列表
        return averages


class TSCEnvWrapper(gym.Wrapper):
    """TSC Env Wrapper for single junction with tls_id
    """
    def __init__(self, env: Env, max_states:int=5, filepath:str=None) -> None:
        super().__init__(env)
        self.tls_ids = self.env.tls_ids # 多路口的 ids
        self.max_states = max_states
        self.states: Dict[str, deque] = {}  # 队列, 记录每个 junction 的 state
        self.occupancy: Dict[str, OccupancyList] = {}  # 计算每个路口的 occupancy
        self.veh_ids: Dict[str, VehicleIDList] = {}
        # (movement_ids & phase2movements) are used for rule-based method
        self.movement_ids: Dict[str, List[str]] = dict()
        self.phase2movements: Dict[str, Dict[str, List[str]]] = dict()
        self.movement2edge: Dict[str, Dict[str, str]] = {}
        self.incoming_edges: Dict[str, List[str]] = {}
        self.edge_to_lanes: Dict[str, Dict[str, List[str]]] = {}
        self.phase2edges: Dict[str, Dict[str, List[str]]] = {}
        self.occupancy_calculator = Occupancy() # 通过车辆尺寸计算占有率
        self.lane_state: Dict[str, Dict[str, Any]] = dict() # 静态 lane 信息缓存
        self.phase_dim: Dict[str, int] = {}
        self.current_phase: Dict[str, int] = {}

        # #######
        # Writer
        # #######
        logger.info(f'RL: Log Path, {filepath}')
        self.t_start = time.time()
        self.results_writer = ResultsWriter(
                filepath,
                header={"t_start": self.t_start},
        )
        self.rewards_writer: List[float] = list()

    @staticmethod
    def _lane_to_edge_id(lane_id: str, lane_state: Dict[str, Dict[str, Any]]) -> Optional[str]:
        """Map lane_id to edge_id using lane_state or lane_id pattern."""
        if not lane_id:
            return None
        lane_info = lane_state.get(lane_id, {})
        edge_id = lane_info.get("edge_id")
        if edge_id:
            return edge_id
        # Fallback: strip lane suffix after the last underscore
        if "_" in lane_id:
            return lane_id.rsplit("_", 1)[0]
        return lane_id

    def get_state(self):
        """将 state 从二维 (max_states, edge_dim) 转换为一维，同时附加当前 phase one-hot."""
        new_state = dict()
        for _tls_id in self.tls_ids:
            occ = np.array(self.states[_tls_id], dtype=np.float32).reshape(-1)
            phase_dim = self.phase_dim.get(_tls_id, 0)
            phase_vec = np.zeros(phase_dim, dtype=np.float32)
            if phase_dim > 0:
                idx = int(self.current_phase.get(_tls_id, 0)) % phase_dim
                phase_vec[idx] = 1.0
            new_state[_tls_id] = {
                "occupancy": occ,
                "phase": phase_vec
            }
        return new_state
    
    # ENV Spaces
    @property
    def action_space(self):
        spaces = {}
        for _tls_id in self.tls_ids:
            phase_dim = self.phase_dim.get(_tls_id, 1)
            # 2 * phase_dim actions: phase_idx in [0, phase_dim-1] for +delta, phase_idx+phase_dim for -delta
            spaces[_tls_id] = gym.spaces.Discrete(max(1, phase_dim * 2))
        return spaces
    
    @property
    def observation_space(self):
        spaces = {}
        for _tls_id in self.tls_ids:
            edge_dim = len(self.incoming_edges.get(_tls_id, [])) or (
                len(self.states[_tls_id][0]) if self.states.get(_tls_id) else 1
            )
            phase_dim = self.phase_dim.get(_tls_id, 0)
            spaces[_tls_id] = gym.spaces.Dict({
                "occupancy": gym.spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.max_states * edge_dim,),
                    dtype=np.float32,
                ),
                "phase": gym.spaces.Box(
                    low=np.zeros(phase_dim, dtype=np.float32),
                    high=np.ones(phase_dim, dtype=np.float32),
                    shape=(phase_dim,),
                    dtype=np.float32,
                )
            })
        return spaces

    # Wrapper
    def state_wrapper(self, tls_id: str, tls_state, vehicle_state, lane_state):
        """返回当前 tls 每个 incoming edge 的 occupancy
        """
        lane_state = lane_state or self.lane_state
        occupancy = self.occupancy_calculator.calculate_edges(
            incoming_edges=self.incoming_edges.get(tls_id, []),
            edge_to_lanes=self.edge_to_lanes.get(tls_id, {}),
            vehicle_state=vehicle_state,
            lane_state=lane_state,
        )
        if not occupancy and tls_state is not None:
            occupancy = tls_state.get("last_step_occupancy", [])
        edge_dim = len(self.incoming_edges.get(tls_id, []))
        if edge_dim and len(occupancy) != edge_dim:
            # pad/truncate to keep shape consistent
            occupancy = list(occupancy)[:edge_dim] + [0.0] * max(0, edge_dim - len(occupancy))
        can_perform_action = tls_state.get('can_perform_action', False) if tls_state else False
        vehicle_ids = tls_state.get('last_step_vehicle_id_list', []) if tls_state else []
        
        return occupancy, vehicle_ids, can_perform_action
    
    # def reward_wrapper(self, vehicle_info, tls_vehicle_ids) -> float:
    #     """返回路口对应的车辆的等待时间, 有可能车辆离开路口, 此时等待时间是 0 (所有车辆的平均等待时间)
    #     """
    #     waiting_times = [min(vehicle_info.get(_veh_id, {}).get('waiting_time', 0), 80) for _veh_id in tls_vehicle_ids]

    #     return -np.mean(waiting_times) if waiting_times else 0
    
    def reward_wrapper(self, vehicle_info, tls_vehicle_ids) -> float:
        # """返回路口对应车辆的平均等待时间的负值（越小越好）"""
        # waiting_times = [min(vehicle_info.get(_veh_id, {}).get('waiting_time', 0), 80) for _veh_id in tls_vehicle_ids]
        # return -float(np.mean(waiting_times)) if waiting_times else 0.0
        # waiting_count = 0
        # for _veh_id in tls_vehicle_ids:
        #     if vehicle_info.get(_veh_id, {}).get('waiting_time', 0) > 0.0: waiting_count += 1
        # return -waiting_count

        waiting_times = 0
        for _veh_id in tls_vehicle_ids:
            waiting_times += vehicle_info.get(_veh_id, {}).get('waiting_time', 0)
        return -waiting_times

    def info_wrapper(self, infos, occupancy, can_perform_action, tls_id):
        """在 info 中加入每个 phase 的占有率 (edge-based)."""
        incoming_edges = self.incoming_edges.get(tls_id, [])
        edge_occ = {key: value for key, value in zip(incoming_edges, occupancy)}
        phase_occ = {}
        for phase_index, edges in self.phase2edges.get(tls_id, {}).items():
            phase_occ[f"{phase_index}"] = sum(edge_occ.get(edge, 0.0) for edge in edges)

        infos = infos or {}
        infos[tls_id] = phase_occ
        infos[tls_id]['can_perform_action'] = can_perform_action
        infos[tls_id]['incoming_edges'] = incoming_edges
        for phase_index, edges in self.phase2edges.get(tls_id, {}).items():
            infos[tls_id][f'green_edges_phase_{phase_index}'] = edges
        return infos

    def reset(self, seed=1) -> Tuple[Any, Dict[str, Any]]:
        """reset 时初始化 (1) 静态信息; (2) 动态信息"""
        state =  self.env.reset()
        self.lane_state = state.get('lane', self.lane_state)
        # 初始化路口静态信息
        for _tls_id in self.tls_ids:
            tls_state = state['tls'][_tls_id]
            self.movement_ids[_tls_id] = tls_state.get('movement_ids', [])
            self.phase2movements[_tls_id] = tls_state.get('phase2movements', {})
            # Build lane/edge mappings
            incoming_edges, edge_to_lanes, movement2edge, phase2edges = self._build_edge_mappings(
                tls_id=_tls_id,
                tls_state=tls_state,
                lane_state=self.lane_state,
            )
            self.incoming_edges[_tls_id] = incoming_edges
            self.edge_to_lanes[_tls_id] = edge_to_lanes
            self.movement2edge[_tls_id] = movement2edge
            self.phase2edges[_tls_id] = phase2edges
            self.phase_dim[_tls_id] = len(self.phase2edges[_tls_id]) if self.phase2edges[_tls_id] else 0
            self.current_phase[_tls_id] = tls_state.get('this_phase_index', 0)

            edge_dim = len(self.incoming_edges[_tls_id])
            self.states[_tls_id] = deque(
                [np.zeros(edge_dim, dtype=np.float32) for _ in range(self.max_states)],
                maxlen=self.max_states,
            )
            self.occupancy[_tls_id] = OccupancyList()
            self.veh_ids[_tls_id] = VehicleIDList()

            # 处理路口动态信息
            occupancy, _, _ = self.state_wrapper(
                tls_id=_tls_id,
                tls_state=tls_state,
                vehicle_state=state.get('vehicle', {}),
                lane_state=self.lane_state
            )
            normalized_occ = (np.array(occupancy, dtype=np.float32) / 100.0).tolist()
            self.states[_tls_id].append(normalized_occ)

        state = self.get_state()
        return state, {'step_time':0}

    def _build_edge_mappings(self, tls_id: str, tls_state: Dict[str, Any], lane_state: Dict[str, Dict[str, Any]]):
        movement_lane_ids = tls_state.get("movement_lane_ids", {})
        movement_ids = tls_state.get("movement_ids", [])
        edge_to_lanes: Dict[str, List[str]] = defaultdict(list)
        movement2edge: Dict[str, str] = {}
        for movement_id in movement_ids:
            lane_ids = movement_lane_ids.get(movement_id, [])
            chosen_edge = None
            for lane_id in lane_ids:
                edge_id = self._lane_to_edge_id(lane_id, lane_state)
                if not edge_id:
                    continue
                edge_to_lanes[edge_id].append(lane_id)
                if chosen_edge is None:
                    chosen_edge = edge_id
            if chosen_edge:
                movement2edge[movement_id] = chosen_edge
        edge_to_lanes = {edge: sorted(set(lanes)) for edge, lanes in edge_to_lanes.items()}
        incoming_edges = sorted(edge_to_lanes.keys())
        if not incoming_edges:
            fallback_occ = tls_state.get("last_step_occupancy", []) or []
            fallback_dim = len(fallback_occ) if fallback_occ else 1
            incoming_edges = [f"{tls_id}_edge_{i}" for i in range(fallback_dim)]
            edge_to_lanes = {edge: [] for edge in incoming_edges}
            logger.warning(f"[{tls_id}] No incoming edges inferred; falling back to {len(incoming_edges)} placeholder edges.")

        phase2edges: Dict[str, List[str]] = {}
        for phase_index, phase_movements in self.phase2movements.get(tls_id, {}).items():
            edge_set = {movement2edge[mv] for mv in phase_movements if mv in movement2edge}
            phase2edges[phase_index] = sorted(edge_set)

        return incoming_edges, edge_to_lanes, movement2edge, phase2edges

    def step(self, action: Dict[str, int]) -> Tuple[Any, SupportsFloat, bool, bool, Dict[str, Any]]:
        can_flags = {_tls_id: False for _tls_id in self.tls_ids}
        while not all(can_flags.values()):
            states, rewards, truncated, done, infos = super().step(action) # 与环境交互
            current_lane_state = states.get('lane', self.lane_state)
            if not self.lane_state:
                self.lane_state = current_lane_state
            for _tls_id in self.tls_ids:
                occupancy, vehicle_ids, can_perform_action = self.state_wrapper(
                    tls_id=_tls_id,
                    tls_state=states['tls'][_tls_id],
                    vehicle_state=states.get('vehicle', {}),
                    lane_state=current_lane_state
                ) # 处理每一帧的数据
                self.occupancy[_tls_id].add_element(occupancy)
                self.veh_ids[_tls_id].add_element(vehicle_ids)
                can_flags[_tls_id] = can_perform_action
                # 更新 phase 信息
                self.phase_dim[_tls_id] = len(self.phase2edges.get(_tls_id, {}))
                self.current_phase[_tls_id] = states['tls'][_tls_id].get('this_phase_index', self.current_phase.get(_tls_id, 0))
        
        # 当可以执行动作的时候, 开始处理时序的 state
        rewards = dict()
        truncateds = dict()
        dones = dict()
        for _tls_id in self.tls_ids:
            avg_occupancy = self.occupancy[_tls_id].calculate_average() # 计算 tls 的 state (0-1)
            rewards[_tls_id] = self.reward_wrapper(
                vehicle_info=states['vehicle'],
                tls_vehicle_ids=self.veh_ids[_tls_id].flatten_remove_duplicates_elements()
            ) # 计算每个 tls 的 vehicle waiting time
            infos = self.info_wrapper(
                infos, occupancy=avg_occupancy, 
                can_perform_action=can_flags[_tls_id],
                tls_id=_tls_id
            ) # info 里面包含每个 phase 的排队
            self.states[_tls_id].append(avg_occupancy.tolist() if hasattr(avg_occupancy, "tolist") else avg_occupancy) # 这里 state 是一个时间序列
            truncateds[_tls_id] = truncated
            dones[_tls_id] = done
        state = self.get_state() # 得到 state
        
        # Writer
        self.rewards_writer.append(float(sum(rewards.values())))
        if all(dones.values()):
            ep_rew = sum(self.rewards_writer)
            ep_len = len(self.rewards_writer)
            ep_info = {"r": round(ep_rew, 6), "l": ep_len, "t": round(time.time() - self.t_start, 6)}
            self.results_writer.write_row(ep_info)
            self.rewards_writer = list()

        return state, rewards, truncateds, dones, infos
    
    def close(self) -> None:
        self.results_writer.close()
        return super().close()


if __name__ == "__main__":
    # Example (pseudo) usage for manual smoke testing:
    # from Env.tsc_env import TSCEnvironment
    # base_env = TSCEnvironment(sumo_cfg="...", net_file="...", trip_info=None,
    #                           num_seconds=100, tls_ids=["J2"], tls_action_type="next_or_not", use_gui=False)
    # env = TSCEnvWrapper(base_env)
    # obs, info = env.reset()
    # print("Incoming edges:", env.incoming_edges)
    # action = {tls_id: 0 for tls_id in env.tls_ids}
    # obs, reward, trunc, done, info = env.step(action)
    pass
