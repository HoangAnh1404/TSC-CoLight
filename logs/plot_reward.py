from pathlib import Path
from tshub.utils.plot_reward_curves import plot_multi_reward_curves
from tshub.utils.get_abs_path import get_abs_path

try:
    current_file = __file__
except NameError:
    current_file = str(Path.cwd() / 'compare_result.ipynb')

path_convert = get_abs_path(current_file)

dirs_and_labels = {
    # 'Choose Next Phase': [
    #     path_convert(f'./TSCRL/result/{env_name}/choose_next_phase/log/{i}.monitor.csv')
    #     for i in range(6)
    # ],
    'Next or Not': [
        path_convert(f'./{i}.monitor.csv')
        for i in range(5)
    ]
}

ouput_file = f'./result.png'

plot_multi_reward_curves(dirs_and_labels, ouput_file)
