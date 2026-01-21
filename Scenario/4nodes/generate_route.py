'''
@Author: WANG Maonan
@Date: 2023-11-23 22:56:22
@Description: 同时生成行人和车辆的 route
@LastEditTime: 2023-11-24 22:23:06
'''
from tshub.utils.get_abs_path import get_abs_path
from tshub.utils.init_log import set_logger
from tshub.sumo_tools.generate_routes import generate_route
import numpy as np

# 初始化日志
current_file_path = get_abs_path(__file__)
set_logger(current_file_path('./'))

# 开启仿真 --> 指定 net 文件
sumo_net = current_file_path("./env/4nodes.net.xml")

interval = [1] * 1000  # 10
slots = len(interval)

def random_turn_ratios(num_slots: int):
    """Generate per-slot right/straight/left ratios that sum to 1 each slot."""
    ratios = np.random.dirichlet([1, 1, 1], size=num_slots)
    return (
        ratios[:, 0].tolist(),  # right turn share per slot
        ratios[:, 1].tolist(),  # straight share per slot
        ratios[:, 2].tolist(),  # left turn share per slot
    )

small_vehicle_num = np.random.randint(3, 5)
medium_vehicle_num = np.random.randint(10, 15)
large_vehicle_num = np.random.randint(30, 40)
vehicle_num = medium_vehicle_num

r_turn, s_turn, l_turn = random_turn_ratios(slots)

generate_route(
    sumo_net=sumo_net,
    interval=interval,
    edge_flow_per_minute={
        'E0': [vehicle_num]*len(interval),
        '-E1': [vehicle_num]*len(interval),
        '-E11': [vehicle_num]*len(interval),
        '-E10': [vehicle_num]*len(interval),
        '-E9': [vehicle_num]*len(interval),
        '-E8': [vehicle_num]*len(interval),
        '-E5': [vehicle_num]*len(interval),
        '-E4': [vehicle_num]*len(interval),


    }, # 每分钟每个入口 edge 的流量
    edge_turndef={
        # J2
        'E0__E1': r_turn,
        'E0__E2': s_turn,
        'E0__E3': l_turn,

        '-E1__E2': r_turn,
        '-E1__E3': s_turn,
        '-E1__-E0': l_turn,

        '-E2__E3': r_turn,
        '-E2__-E0': s_turn,
        '-E2__E1': l_turn,

        '-E3__-E0': r_turn,
        '-E3__E1': s_turn,
        '-E3__E2': l_turn,

        # J4
        'E2__E11': r_turn,
        'E2__E10': s_turn,
        'E2__-E7': l_turn,

        '-E11__E10': r_turn,
        '-E11__-E7': s_turn,
        '-E11__-E2': l_turn,

        '-E10__-E7': r_turn,
        '-E10__-E2': s_turn,
        '-E10__E11': l_turn,

        'E7__-E2': r_turn,
        'E7__E11': s_turn,
        'E7__E10': l_turn,

        #  J5
        'E3__E6': r_turn,
        'E3__E5': s_turn,
        'E3__E4': l_turn,
        
        '-E6__E5': r_turn,
        '-E6__E4': s_turn,
        '-E6__-E3': l_turn,

        '-E5__E4': r_turn,
        '-E5__-E3': s_turn,
        '-E5__E6': l_turn,
        
        '-E4__-E3': r_turn,
        '-E4__E6': s_turn,
        '-E4__E5': l_turn,

        # J8
        '-E7__E9': r_turn,
        '-E7__E8': s_turn,
        '-E7__-E6': l_turn,
        
        '-E9__E8': r_turn,
        '-E9__-E6': s_turn,
        '-E9__E7': l_turn,
        
        '-E8__-E6': r_turn,
        '-E8__E7': s_turn,
        '-E8__E9': l_turn,
        
        'E6__E7': r_turn,
        'E6__E9': s_turn,
        'E6__E8': l_turn,

    },
    veh_type={
        'motorbike': {'color':'yellow', 'vClass':'motorcycle', 'length':2.00, 'minGap':0.5, 'accel':1.5, 'maxSpeed':11.11, 'departSpeed':11.11, 'desiredMaxSpeed':11.11, 'probability':0.7},
        'car': {'color':'155, 89, 182', 'vClass':'passenger', 'length':4.50, 'minGap':1.0, 'accel':1.5, 'maxSpeed':11.11, 'departSpeed':11.11, 'desiredMaxSpeed':11.11, 'probability':0.3},
    },

    walkfactor=0.5,
    output_trip=current_file_path('./_testflow.trip.xml'),
    output_turndef=current_file_path('./_testflow.turndefs.xml'),
    output_route=current_file_path('./routes/vehicle.rou.xml'),
    # person_trip_file=current_file_path('./_pedestrian.trip.xml'),
    # output_person_file=current_file_path('./routes/gio_cao_diem/pedestrian.rou.xml'),
    interpolate_flow=False,
    interpolate_turndef=False,
    interpolate_walkflow=False
)
