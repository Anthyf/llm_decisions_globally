#!/venv_llm_trade_offs/bin/python3
"""
Script to plot the results of the experiments.
"""
import pickle
import copy
from utils.utils import LR_transition_values, LR_values_per_quantity_table
from pathlib import Path


def tradeoff_table(args):
    """
    Generate LLM trade‑off tables for *all* experiments listed in
    args["trade_off_values"].
    
    Before run this script, make sure:
    Step 1. to validate the "experiment_name" in config_file
    Step 2. to validate the money_min, money_max and log scale suits our needs. 
    Step 3. keep there is only one file under each experiment_name folder.

    """
    print("\nArguments in use:")
    for k, v in args.items():
        if k not in (
            "experiment_name",  # changing experiment_name in config_file
            "money_min",        # fixed log scale 0.1 dollar 
            "money_max",        # fixed log scale 100 dollar 
            "quantity_min",     # fixed 60min
            "quantity_max",     # fixed, but no use
            "log_scale_money",  # fixed, True, by default
        ):
            continue
        print(f"\t{k}: {v}")

    exp_map = args["trade_off_values"] # now, to validate if config_file-->trade_off_values is our target file.
    for k, v in exp_map.items():
        print(f"\ttrade_off_values: {k}")
        for k1, v1 in v.items():
            print(f"\t\t{k1}: {v1}")

    experiment_names = set(exp_map.keys()) # a set of experiment names in config_file under "trade_off_values", e.g. {"test2_countrytxt", "test3_countrytxt", ...}
    data_files = []

    # Sanity and data-gathering check: look for the file make sure it exists, and gather all the data files to be processed.
    root = Path(args["output_dir"])
    for p in root.rglob("data*.pkl"): #A recursive glob looks inside the targeted folder, then looks inside every folder inside that folder...; * means any characters, so it will match both "data.pkl" and "data_log.pkl", etc.
        if p.name not in {"data.pkl", "data_log.pkl"}:
            continue

        exp_name = p.parent.name  # …/<prompt_type>/<experiment_name>/data_log.pkl, so p.parent is the folder of experiment_name, and p.parent.name is the name of that folder, which is the experiment_name.
        if exp_name not in experiment_names:
            continue

        prompt_type = p.parent.parent.name  # …/<prompt_type>/<experiment_name>/data_log.pkl, so p.parent.parent is the folder of prompt_type, and p.parent.parent.name is the name of that folder, which is the prompt_type.
        data_files.append((p, exp_name, prompt_type, p.name == "data_log.pkl"))

    if not data_files:
        print("\nNo data.pkl or data_log.pkl files found under", root)
        return

    for path, experiment_name, prompt_type, is_log in data_files:
        run_args = copy.deepcopy(args)
        run_args["experiment_name"] = experiment_name
        run_args["prompt_type"] = prompt_type
        run_args["log_scale_money"] = is_log
        print("*" * 70)
        print(f"Loading data from: {path}")

        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            print(f"\tFailed to load '{path}': {e}")
            continue

        for model, df in data.items():
            if df is None:
                print(f"\tSkipping {model}: no valid data.")
                continue
            LR_transition_values(run_args, data, model)

    # LR_values_per_quantity_table(args)
