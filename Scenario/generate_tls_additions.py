'''
@Author: WANG Maonan
@Date: 2023-09-01 15:42:25
@Description: 生成 Traffic Light 的 add 文件
@LastEditTime: 2023-09-01 15:52:52
'''
from tshub.sumo_tools.additional_files.traffic_light_additions import generate_traffic_lights_additions
from tshub.utils.get_abs_path import get_abs_path
from tshub.utils.init_log import set_logger

# 初始化日志
current_file_path = get_abs_path(__file__)
set_logger(current_file_path('./'))

env_name = '4nodes'

# 指定 net 文件
sumo_net = current_file_path(f"./{env_name}/env/{env_name}.net.xml")

generate_traffic_lights_additions(
    network_file=sumo_net,
    output_file=current_file_path(f'./{env_name}/add/tls.add.xml')
)