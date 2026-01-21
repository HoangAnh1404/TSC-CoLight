from typing import List
from Env.tsc_env import TSCEnvironment
from Env.tsc_wrapper import TSCEnvWrapper

def make_multi_envs(
        tls_ids: List[str], sumo_cfg:str,
        num_seconds: int, use_gui: bool,
        net_file: str, trip_info: str,
        tls_action_type: str,
        log_path: str,
    ):
    tsc_env = TSCEnvironment(sumo_cfg, net_file, trip_info, num_seconds, tls_ids, tls_action_type, use_gui)
    tsc_env = TSCEnvWrapper(tsc_env, max_states=5, filepath=log_path)

    return tsc_env
