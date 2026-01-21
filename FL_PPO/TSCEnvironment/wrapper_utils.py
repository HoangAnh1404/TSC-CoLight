'''
@Author: WANG Maonan
@Date: 2023-09-05 15:26:11
@Description: 处理 State 的特征
@LastEditTime: 2023-12-02 17:35:36
'''
import os
import xml.etree.ElementTree as ET

import sumolib

import numpy as np
from typing import List, Dict, Any, Tuple, Set
from TransSimHub.tshub.utils.nested_dict_conversion import create_nested_defaultdict, defaultdict2dict

class OccupancyList:
    def __init__(self) -> None:
        self.elements = []

    def add_element(self, element) -> None:
        if isinstance(element, list):
            if all(isinstance(e, float) for e in element):
                self.elements.append(element)
            else:
                raise ValueError("列表中的元素必须是浮点数类型")
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

def find_index(lst, element):
    try:
        return lst.index(element)
    except ValueError:
        return None


def calculate_queue_lengths(movement_ids:List[str], jam_length_meters:List[float], phase2movements:Dict[str, List[str]]) -> Dict[str, Dict[str, float]]:
    """计算每个相位的平均和最大排队长度

    Args:
        movement_ids (List[str]): 路口 movement 的顺序
            movement_ids = [
                "161701303#7.248_l", "161701303#7.248_r", "161701303#7.248_s",
                "29257863#2_l", "29257863#2_r", "29257863#2_s",
                "gsndj_n7_l", "gsndj_n7_r", "gsndj_n7_s",
                "gsndj_s4_l", "gsndj_s4_r", "gsndj_s4_s"
            ]
        jam_length_meters (List[float]): 每个 movement 对应的排队长度, 与上面的顺序相同
            jam_length_meters = [
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                0.0, 0.0, 60.83249079171935,
                0.0, 0.0, 68.70503137164724
            ]
        phase2movements (Dict[str, List[str]]): 每个 phase 包含的 movement id
            phase2movements = {
                "0": [
                    "gsndj_s4--r",
                    "gsndj_s4--s",
                    "gsndj_n7--s",
                    "gsndj_n7--r"
                ],
                "1": [
                    "gsndj_s4--l",
                    "gsndj_n7--l"
                ],
                "2": [
                    "29257863#2--s",
                    "29257863#2--r",
                    "161701303#7.248--r",
                    "161701303#7.248--s"
                ],
                "3": [
                    "161701303#7.248--l",
                    "29257863#2--l"
                ]
            }
    Returns:
        Dict[str, Dict[str, float]]: 计算每一个 phase 的最大和平均排队长度
            {
                0: {'total_length': 0.0, 'count': 4, 'max_length': 0.0, 'average_length': 0.0}, 
                1: {'total_length': 0.0, 'count': 2, 'max_length': 0.0, 'average_length': 0.0}, 
                2: {'total_length': 0.0, 'count': 4, 'max_length': 0.0, 'average_length': 0.0}, 
                3: {'total_length': 0.0, 'count': 2, 'max_length': 0.0, 'average_length': 0.0}
            }
    """
    phase_queue_lengths = {}

    # 初始化每个 phase 的总排队长度和计数器
    for phase in phase2movements:
        phase_queue_lengths[phase] = {
            'total_length': 0.0,
            'count': 0,
            'max_length': 0.0,
            'average_length': 0.0
        }

    # 遍历每个 phase，累加每个 movement 的排队长度
    for phase, movements in phase2movements.items():
        for movement in movements:
            movement = '_'.join(movement.split('--'))
            index = movement_ids.index(movement)
            length = jam_length_meters[index]
            phase_queue_lengths[phase]['total_length'] += length
            phase_queue_lengths[phase]['count'] += 1
            phase_queue_lengths[phase]['max_length'] = max(phase_queue_lengths[phase]['max_length'], length)

    # 计算每个 phase 的平均排队长度
    for phase, data in phase_queue_lengths.items():
        if data['count'] > 0:
            data['average_length'] = data['total_length'] / data['count']

    return phase_queue_lengths


def predict_queue_length(queue_info:Dict[str, float], is_green:bool=False, num_samples = 10):
    leaving_rate_lambda = 4 # 离开率的参数 λ
    predict_queue_info = {} # 预测的排队长度
    for _id, _queue_length in queue_info.items():
        if _id == 'max_length':
            arrival_rate_lambda = 3 # 到达率的参数 λ
        elif _id == 'average_length':
            arrival_rate_lambda = 2 # 到达率的参数 λ
        else:
            continue
            
        if is_green:
            sample_sum = 0
            for _ in range(num_samples):
                sample_sum += np.random.poisson(arrival_rate_lambda) - np.random.poisson(leaving_rate_lambda)
            sample_sum *= 6 # 车辆数 --> 排队长度
            predicted_length = max(_queue_length + sample_sum / num_samples, 0)
            predict_queue_info[_id] = predicted_length
        else:
            sample_sum = 0
            for _ in range(num_samples):
                sample_sum += np.random.poisson(arrival_rate_lambda)
            sample_sum *= 6 # 车辆数 --> 排队长度
            predicted_length = max(_queue_length + sample_sum / num_samples, 0)
            predict_queue_info[_id] = predicted_length
    return predict_queue_info


def convert_state_to_static_information(input_data) -> Dict[str, Dict[str, Any]]:
    """将 state 输出为路网的静态信息

    Args:
        input_data: 单个 Traffic Light 的 state. 
        {
            'movement_directions': {'E2_r': 'r', 'E2_s': 's', ...},
            'movement_ids': ['E2_l', 'E2_r', 'E2_s', 'E4_l', ...],
            'phase2movements': {0: ['E2--s', 'E1--s'], 1: ['E1--l', 'E2--l'], ...},
            'movement_lane_numbers': {'-E2_r': 1, '-E2_s': 1, '-E2_l': 1, ...}
        }

    Returns:
        Dict[str, Dict[str, Any]]: 将其转换为路口的静态信息
        {
            "movement_infos": {
                "E2_l": {
                    "direction": "Left Turn",
                    "number_of_lanes": 1
                },
                "E2_s": {
                    "direction": "Through",
                    "number_of_lanes": 1
                },
                ...
            },
            "phase_infos": {
                "phase 0": {
                    "movements": ["E2--s", "E1--s"]
                },
                "phase 1": {
                    "movements": ["E1--l", "E2--l"]
                },
                ...
            }
        }
    """
    output_data = {
        "movement_infos": {},
        "phase_infos": {}
    }

    # 处理 movement_directions
    for movement_id, direction in input_data["movement_directions"].items():
        if direction == "l":
            direction_text = "Left Turn"
        elif direction == "s":
            direction_text = "Through"
        else:
            continue

        number_of_lanes = input_data["movement_lane_numbers"].get(movement_id, 0)

        output_data["movement_infos"][movement_id] = {
            "direction": direction_text,
            "number_of_lanes": number_of_lanes
        }

    # 处理 phase2movements
    for phase, movements in input_data["phase2movements"].items():
        phase_key = f"Phase {phase}"
        output_data["phase_infos"][phase_key] = {
            "movements": movements
        }

    return output_data

def _get_net_file_from_sumo_cfg(sumo_cfg: str) -> str:

    if not os.path.exists(sumo_cfg):
        raise FileNotFoundError(f"SUMO config file not found: {sumo_cfg}")

    tree = ET.parse(sumo_cfg)
    root = tree.getroot()

    # SUMO thường khai báo:
    # <input>
    #   <net-file value="my_network.net.xml"/>
    #   ...
    # </input>
    net_elem = root.find(".//net-file")
    if net_elem is None or "value" not in net_elem.attrib:
        raise ValueError(
            f"Cannot find <net-file> element in SUMO cfg: {sumo_cfg}"
        )

    net_path = net_elem.get("value")
    # Nếu là path tương đối → join với thư mục chứa sumo_cfg
    if not os.path.isabs(net_path):
        net_path = os.path.join(os.path.dirname(sumo_cfg), net_path)

    if not os.path.exists(net_path):
        raise FileNotFoundError(
            f"Network file from cfg not found: {net_path}"
        )

    return net_path


def get_neighbors(net_file: str, tls_id: str) -> List[str]:
    """Lấy danh sách tls láng giềng trực tiếp của một junction có đèn tín hiệu.

    Args:
        net_file (str): Đường dẫn tới file .net.xml.
        tls_id (str): Id junction trung tâm (kiểu traffic_light).

    Returns:
        List[str]: Danh sách id các traffic light kề với tls_id qua một edge.
    """
    if not os.path.exists(net_file):
        raise FileNotFoundError(f"Network file not found: {net_file}")

    net = sumolib.net.readNet(net_file)
    tls_nodes = {tl.getID() for tl in net.getTrafficLights()}
    if tls_id not in tls_nodes:
        raise ValueError(f"tls_id '{tls_id}' is not a traffic light in net '{net_file}'")

    neighbors: Set[str] = set()
    for edge in net.getEdges():
        from_node = edge.getFromNode()
        to_node = edge.getToNode()
        if from_node is None or to_node is None:
            continue

        f_id = from_node.getID()
        t_id = to_node.getID()

        if f_id == tls_id and t_id in tls_nodes:
            neighbors.add(t_id)
        elif t_id == tls_id and f_id in tls_nodes:
            neighbors.add(f_id)

    return sorted(neighbors)

def build_adjacency_from_sumo(sumo_cfg: str, tls_ids):

    net_file = _get_net_file_from_sumo_cfg(sumo_cfg)

    neighbors = {tls: set() for tls in tls_ids}
    tls_set = set(tls_ids)

    net = sumolib.net.readNet(net_file)
    for edge in net.getEdges():
        from_node = edge.getFromNode()
        to_node = edge.getToNode()
        if from_node is None or to_node is None:
            continue

        f_id = from_node.getID()
        t_id = to_node.getID()
        if f_id in tls_set and t_id in tls_set:
            neighbors[f_id].add(t_id)
            neighbors[t_id].add(f_id)

    # Trả về dict {tls_id: [neighbor1, neighbor2, ...]}
    return {k: list(v) for k, v in neighbors.items()}

def build_local_graphs(
    adj_dict: Dict[str, List[str]],
    max_neighbors: int = 4,
    dummy_prefix: str = "__dummy__",
) -> Tuple[Dict[str, List[str]], Dict[str, np.ndarray]]:

    local_nodes_map: Dict[str, List[str]] = {}
    adj_local_map: Dict[str, np.ndarray] = {}

    for center, neighbors in adj_dict.items():
        # Bỏ chính nó nếu lỡ có trong list neighbors
        uniq_neighbors = [n for n in neighbors if n != center]

        # Giữ thứ tự, loại trùng
        seen = set()
        filtered = []
        for n in uniq_neighbors:
            if n not in seen:
                seen.add(n)
                filtered.append(n)

        # Cắt theo max_neighbors
        filtered = filtered[:max_neighbors]

        # Padding bằng dummy node nếu thiếu
        num_dummy = max_neighbors - len(filtered)
        dummy_nodes = [f"{dummy_prefix}{center}_{i}" for i in range(num_dummy)]

        # Node list: [center] + neighbors + dummies
        nodes = [center] + filtered + dummy_nodes
        local_nodes_map[center] = nodes

        # Xây adjacency cục bộ (M x M)
        M = len(nodes)
        A = np.zeros((M, M), dtype=np.float32)

        # Map các node thật (không phải dummy) sang index
        node_index = {
            nid: idx
            for idx, nid in enumerate(nodes)
            if not nid.startswith(dummy_prefix)
        }

        # Fill adjacency cho subgraph
        for nid_i, idx_i in node_index.items():
            for nei in adj_dict.get(nid_i, []):
                idx_j = node_index.get(nei)
                if idx_j is not None:
                    A[idx_i, idx_j] = 1.0
                    A[idx_j, idx_i] = 1.0  # cho đơn giản: coi như undirected

        adj_local_map[center] = A

    return local_nodes_map, adj_local_map


def get_incoming_lane_count(sumo_cfg: str, junction_id: str) -> int:

    net_file = _get_net_file_from_sumo_cfg(sumo_cfg)
    net = sumolib.net.readNet(net_file)
    node = net.getNode(junction_id)
    if node is None:
        raise ValueError(f"Junction id '{junction_id}' not found in net file '{net_file}'")

    total_lanes = 0
    for edge in net.getEdges():
        to_node = edge.getToNode()
        if to_node is None:
            continue
        if to_node.getID() == junction_id:
            lanes = edge.getLanes()
            total_lanes += len(lanes)

    return total_lanes
