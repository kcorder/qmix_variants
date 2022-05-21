import json
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pprint import pprint
from collections import defaultdict
import logging
import seaborn as sns

sns.set()
colors = sns.color_palette()


parser = argparse.ArgumentParser()
parser.add_argument("--map-name", type=str, required=True,
                    choices=["3m", "2c_vs_64zg", "2s3z", "3s_vs_5z", "corridor",
                             "simple_reference", "simple_spread"])
parser.add_argument("--metric", type=str, required=True,
                    choices=["test_return_mean", "test_battle_won_mean"])
parser.add_argument("--save-fig", action='store_true', default=False)
parser.add_argument("--legend", default="auto", type=str,
                    choices=["brief","full","auto","False"])
parser.add_argument("--legend-size", type=int)
parser.add_argument("--legend-loc", type=str)
args = parser.parse_args()

if args.legend == "False":
    args.legend = False


metric_to_name = {
    "test_return_mean": "Eval episode reward",
    "test_battle_won_mean": "Eval Win %",
}
alg_to_name = {
    "qmix": "QMIX PS RNN",
    "qmix_fc_nops": "QMIX FC",
    "qmix_nops": "QMIX RNN",
    "qmix_fc": "QMIX PS FC",
    "vdn": "VDN PS RNN",
    "vdn_fc_nops": "VDN FC",
    "vdn_fc": "VDN PS FC",
    "vdn_nops": "VDN RNN",
    "facmac": "FACMAC PS RNN",
    "facmac_fc_nops": "FACMAC FC",
    "facmac_fc": "FACMAC PS FC",
    "facmac_nops": "FACMAC RNN",
}


# Root results dir
runs_dir = '/home/kcorder/Desktop/pymarl_onyx_results'
os.chdir(runs_dir)


# Get all Sacred run directories
sacred_dirs = [d for d in glob.glob("*/*/sacred/*") if not d.endswith("_sources")]

run_configs = {}
run_infos = {}
for d in sacred_dirs:
    assert d[-1].isdigit(), "All dirs here should be run #s"
    try:
        with open(f"{d}/info.json", 'rb') as fd:
            run_infos[d] = json.load(fd)
        with open(f"{d}/config.json", 'rb') as fd:
            run_configs[d] = json.load(fd)
    except (FileNotFoundError, json.decoder.JSONDecodeError) as e:
        sacred_dirs.remove(d)
        run_infos.pop(d, None)
        run_configs.pop(d, None)

run2map = {d: config['env_args']['map_name'] for d, config in run_configs.items()}

map2runs = {}
for key, value in run2map.items():
    map2runs.setdefault(value, [])
    map2runs[value].append(key)

print("\n\n\n")
print("Map to runs:")
pprint(map2runs)



df_dict = defaultdict(list)
map = args.map_name
runs = map2runs[map]
metric = args.metric
alg_runs_per_map = defaultdict(int)  # counting how many alg runs are present for the current env
for run in runs:
    alg_name = run.split('/', maxsplit=1)[0]
    info = run_infos[run]
    config = run_configs[run]
    alg_runs_per_map[alg_name] += 1

    y = [datum['value'] if isinstance(datum, dict) else datum
         for datum in info[metric]]
    df_dict[metric_to_name[metric]].extend(y)
    df_dict['Algorithm'].extend([alg_to_name[alg_name]] * len(y))

    X_MAX = config['t_max'] # - config['test_interval']
    x_range = np.arange(0, X_MAX, config['test_interval'])
    if len(y) < len(x_range):
        logging.warning(f"Run {run} has incomplete data (environment {map}). Expected length {len(x_range)}, got {len(y)}.")
    x_range = x_range[:len(y)]
    df_dict['Environment steps'].extend(x_range)

print(f"\n\n")
print(f"Alg runs for map {map}:")
pprint(alg_runs_per_map)



df = pd.DataFrame(df_dict)

axes = sns.lineplot(data=df, x='Environment steps', y=metric_to_name[metric],
                    hue='Algorithm', linewidth=0.7, legend=args.legend)
if args.legend_size:
    plt.legend(loc=args.legend_loc or "best",
               prop={'size':args.legend_size})

if args.save_fig:
    plt.savefig(f"{args.map_name}_{metric}.pdf", bbox_inches="tight")
else:
    plt.show()



pass

if __name__ == '__main__':

    a=0