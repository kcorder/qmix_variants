import argparse
import itertools
from typing import List


def get_run_cmds(config: dict) -> List[str]:
    """returns list of CLI args as product of config dict options"""
    config_strs = [x for x in itertools.product(*config.values())] # list of lists of args

    # remove empty config terms created by product()
    for i, config_str_list in enumerate(config_strs):
        config_strs[i] = [term for term in config_str_list if term]

    # convert to list of strs
    return [' '.join(x) for x in config_strs]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-seeds", type=int, default=5)
    parser.add_argument("--filename", type=str, default="qmix_runs.txt")
    parser.add_argument("--alg-config", type=str, default=None,
                        help="Which algorithm config file to run. No value runs all alg configs.")
    parser.add_argument("--env", type=str, default="env",
                        help="Which env or scenario to run. No value runs all envs/scenarios.")
    parser.add_argument("--main-file", type=str, default="src/main.py",
                        help="where to find main file relative to CWD")
    args = parser.parse_args()

    global_options = ""
    config_dict = {
        "map": [env_config for env_config in [
            "--env-config=sc2 with env_args.map_name=3m",
            "--env-config=sc2 with env_args.map_name=2s3z",
            "--env-config=sc2 with env_args.map_name=3s_vs_5z",
            "--env-config=sc2 with env_args.map_name=corridor",
            "--env-config=sc2 with env_args.map_name=2c_vs_64zg",
            "--env-config=mpe with env_args.map_name=simple_spread",
            "--env-config=mpe with env_args.map_name=simple_reference",
        ] if args.env in env_config],
        "config": [
            "--config=qmix",
            "--config=qmix_fc",
            "--config=qmix_nops",
            "--config=qmix_fc_nops",
            "--config=vdn",
            "--config=facmac",
            "--config=facmac_fc",
            "--config=facmac_nops",
            "--config=facmac_fc_nops",
        ] if not args.alg_config else [f"--config={args.alg_config}"],
        "seed": ["" for seed in range(args.num_seeds)],
    }



    config_strs = get_run_cmds(config_dict)
    print(*config_strs, sep='\n')
    print("num unique configs =", len(config_strs))
    run_strs = []
    exe_str = f"python -O {args.main_file}"
    for i, args_str in enumerate(config_strs):
        run_strs.append(f"{exe_str} {global_options} {args_str} \n")

    # write file with commands
    with open(args.filename, 'w') as fd:
        fd.writelines(run_strs)
