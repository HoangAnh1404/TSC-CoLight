'''
@Author: WANG Maonan
@Date: 2023-09-08 17:45:54
@Description: 创建 TSC Env + Wrapper
@LastEditTime: 2023-12-02 11:47:12
'''
import sys
import os
from pathlib import Path

parent_directory = Path(__file__).resolve().parent.parent.parent
if str(parent_directory) not in sys.path:
    sys.path.insert(0, str(parent_directory))

import gymnasium as gym

from typing import List
from stable_baselines3.common.monitor import Monitor
from TransSimHub.tshub.tshub_env.base_sumo_env import BaseSumoEnvironment

from FL_PPO.TSCEnvironment.tsc_env import TSCEnvironment
from FL_PPO.TSCEnvironment.base_tsc_wrapper import BaseTSCEnvWrapper

def make_env(
        tls_id:str,
        num_seconds:int,
        phase_num:int, # phase 的数量, 用于初始化 action space
        sumo_cfg:str,
        net_file:str,
        use_gui:bool,
        log_file:str, 
        env_index:int,
        tls_action_type:str='choose_next_phase',
        trip_info:str=None,
        copy_files:List[str]=[],
        label_prefix:str="fl",
    ):
    def _init() -> gym.Env: 
        # Set a unique label prefix per env process to avoid TraCI label/port collisions.
        unique_label_seed = f"{label_prefix}-{tls_id}-{env_index}-{os.getpid()}"
        BaseSumoEnvironment.CONNECTION_LABEL = abs(hash(unique_label_seed)) % 1_000_000_000

        tsc_scenario = TSCEnvironment(
            sumo_cfg=sumo_cfg, 
            net_file=net_file,
            trip_info=trip_info,
            num_seconds=num_seconds,
            tls_id=tls_id, 
            tls_action_type=tls_action_type,
            use_gui=use_gui,
        )
        tsc_wrapper = BaseTSCEnvWrapper(
            tsc_scenario, phase_num=phase_num, 
            tls_id=tls_id, copy_files=copy_files
        )
        return Monitor(tsc_wrapper, filename=f'{log_file}/{env_index}')
    
    return _init
