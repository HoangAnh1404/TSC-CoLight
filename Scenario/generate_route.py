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

env_name = 'test'

# 开启仿真 --> 指定 net 文件
sumo_net = current_file_path(f"./{env_name}/env/{env_name}.net.xml")

interval = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # 10
slots = len(interval)

def repeat(value: float):
    return [value] * slots

small_vehicle_num = np.random.randint(3, 5)
medium_vehicle_num = np.random.randint(10, 15)
large_vehicle_num = np.random.randint(30, 40)
vehicle_num = 25

r, s, l = np.random.dirichlet([1, 1, 1])

generate_route(
    sumo_net=sumo_net,
    interval=interval,
    edge_flow_per_minute={
        '-E1': [vehicle_num]*len(interval),
        '-E2': [vehicle_num]*len(interval),
        '-E4': [vehicle_num]*len(interval),
        'E0': [vehicle_num]*len(interval),
        '-E5': [vehicle_num]*len(interval),
        '-E6': [vehicle_num]*len(interval),
    }, # 每分钟每个入口 edge 的流量
    edge_turndef={
        # J1
        'E0__E1': repeat(r),
        'E0__E2': repeat(s),
        'E0__E3': repeat(l),

        '-E1__E2': repeat(r),
        '-E1__E3': repeat(s),
        '-E1__-E0': repeat(l),

        '-E2__E3': repeat(r),
        '-E2__-E0': repeat(s),
        '-E2__E1': repeat(l),

        '-E3__-E0': repeat(r),
        '-E3__E1': repeat(s),
        '-E3__E2': repeat(l),

        # J4
        'E3__E4': repeat(r),
        'E3__E5': repeat(s),
        'E3__E6': repeat(l),

        '-E4__E5': repeat(r),
        '-E4__E6': repeat(s),
        '-E4__-E4': repeat(l),

        '-E5__E6': repeat(r),
        '-E5__-E3': repeat(s),
        '-E5__E4': repeat(l),

        '-E6__-E3': repeat(r),
        '-E6__E4': repeat(s),
        '-E6__E5': repeat(l),

    },
    veh_type={
        'motorbike': {'color':'yellow', 'vClass':'motorcycle', 'length':2.00, 'minGap':0.5, 'accel':1.5, 'maxSpeed':11.11, 'departSpeed':11.11, 'desiredMaxSpeed':11.11, 'probability':0.7},
        'car': {'color':'155, 89, 182', 'vClass':'passenger', 'length':4.50, 'minGap':1.0, 'accel':1.5, 'maxSpeed':11.11, 'departSpeed':11.11, 'desiredMaxSpeed':11.11, 'probability':0.3},
    },

    walkfactor=0.5,
    output_trip=current_file_path(f'./{env_name}/_testflow.trip.xml'),
    output_turndef=current_file_path(f'./{env_name}/_testflow.turndefs.xml'),
    output_route=current_file_path(f'./{env_name}/routes/vehicle.rou.xml'),
    interpolate_flow=False,
    interpolate_turndef=False,
    interpolate_walkflow=False
)
