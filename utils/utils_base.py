"""
Utils for LLM trade-offs scripts. API KEYS are removed.
"""

from importlib.resources import path
import os
from platform import node
import numpy as np
import pandas as pd
import math
import yaml
import pickle
import time
import random
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.image as mpimg
from pathlib import Path
from functools import reduce
from requests.exceptions import RequestException
from urllib3.exceptions import MaxRetryError, NewConnectionError
from sklearn.linear_model import LogisticRegression
from utils.LLMwrapper import *
import asyncio



def load_config(path: Path = Path("config/config_file.yaml")) -> dict:
    """
    Load and validate configuration from a single YAML file.
    """
    _TYPE_MAP = {
        "str": str,
        "int": int,
        "float": float,
    }

    def country_list_constructor(Loader, node):
        full_path = path.parent / node.value
        
        with open(full_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]

    yaml.SafeLoader.add_constructor('!include', country_list_constructor)
    
    # Leave this exactly as it was:
    with path.open(encoding='utf-8') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    args = {}

    for name, opts in cfg.get("args", {}).items():
        val = opts.get("value")

        if opts.get("required", False) and (
            val is None or val == "" or (isinstance(val, list) and not val)
        ):
            raise ValueError(
                f"'{name}' is required but no value was set in config/config_file.yaml"
            )

        if (val is None or (isinstance(val, list) and not val)) and "default" in opts:
            val = opts["default"]

        if "choices" in opts:
            if opts.get("nargs") == "+":
                if not isinstance(val, list):
                    raise ValueError(f"`{name}` expects a list but got {val!r}")
                invalid = [v for v in val if v not in opts["choices"]]
                if invalid:
                    raise ValueError(
                        f"`{name}` has invalid entries {invalid}; must be one of {opts['choices']}"
                    )
            else:
                if val not in opts["choices"]:
                    raise ValueError(
                        f"`{name}` must be one of {opts['choices']}, got {val!r}"
                    )

        if opts.get("action") == "store_true":
            val = bool(val)

        if "type" in opts and opts.get("action") != "store_true":
            caster = _TYPE_MAP.get(opts["type"])
            if caster:
                if opts.get("nargs") == "+":
                    val = [caster(item) for item in val]
                else:
                    val = caster(val)

        args[name] = val

    prompts = cfg.get("prompts", {})
    base_key = args.get("prompt")
    full_key = f"{base_key}"
    if full_key not in prompts:
        raise KeyError(
            f"Prompt template '{full_key}' not found in config/config_file.yaml"
        )
    prompt_template = prompts[full_key]

    args["api_keys"] = cfg.get("api_keys", {})
    args["x_labels"] = cfg.get("x_labels", {})
    args["prompt_type"] = base_key
    args["prompt"] = prompt_template
    args["semaphore"]= cfg.get("semaphore", {}).get("value") 

    return args


def money_quantity_trade_off(args):
    np.set_printoptions(suppress=True)

    # floating numerical experiment
    if args.get("money_custom_values"):
        money_array = np.array(args["money_custom_values"], dtype=float)
    else:
        money_array = np.linspace(args["money_min"], args["money_max"], args["money_n"])
        if args["log_scale_money"]:
            money_array = 10**money_array

    quant_array = np.linspace(
        args["quantity_min"], args["quantity_max"], args["quantity_n"]
    )

    print(money_array)
    print(quant_array)
    return money_array, quant_array
    

def compute_experiments(args, model, money_array, quant_array, output_file=None, experiment_outcomes=None, model_name=None):
    # Retry configuration
    max_retries = 5
    base_delay = 2
    max_delay = 120
    max_none_per_experiment = 7
    semaphore_value = args.get("semaphore")

    results_list = []
    completed_experiments = set()  # experiment numbers (1-indexed) already done

    expected_rows = len(quant_array) * len(money_array)

    if output_file is not None and model_name is not None and os.path.exists(output_file):
        with open(output_file, "rb") as f:
            loaded_outcomes = pickle.load(f)
        if model_name in loaded_outcomes:
            existing_df = loaded_outcomes[model_name]
            results_list = existing_df.to_dict("records")

            for ep_val, group in existing_df.groupby("experiment"):
                if len(group) >= expected_rows:
                    completed_experiments.add(ep_val)

            print(f"[INFO] Resuming: found {len(results_list)} existing rows, "
                  f"experiments already complete: {sorted(completed_experiments)}")
    else:
        loaded_outcomes = experiment_outcomes if experiment_outcomes is not None else {}

    if experiment_outcomes is None:
        experiment_outcomes = {}
    # Merge any pre-existing outcomes for other models so we don't clobber them on save
    if output_file is not None and os.path.exists(output_file):
        experiment_outcomes = loaded_outcomes

    # We wrap the core logic in an async function so we can manage the event loop
    async def _run_experiments_async():
        # Semaphore limits concurrent requests to prevent HTTP 429 (Too Many Requests) errors.
        semaphore = asyncio.Semaphore(semaphore_value)

        async def _fetch_and_record(ep, quant_value, money_value):
            prompt = generate_prompt(args, money_value, quant_value)
            context_str = f"money_value={money_value}"

            async with semaphore:
                retry_count = 0
                while retry_count <= max_retries:
                    try:
                        response = await model.generate_response(prompt=prompt)
                        break
                    except Exception as e:
                        retry_count += 1
                        if retry_count > max_retries:
                            print(f"[INFO: Return None] {context_str} | Fatal error after {max_retries} retries.")
                            print(f"\t{e}")
                            return {
                                "reward_value": round(float(money_value), 6),
                                "quantity": round(float(quant_value), 6),
                                "experiment": ep + 1,
                                "output": None,
                            }

                        delay = min(base_delay * (2 ** (retry_count - 1)) + random.uniform(0, 1), max_delay)
                        print(f"[RETRY] {context_str} | attempt {retry_count}/{max_retries} failed "
                              f"({type(e).__name__}: {str(e)[:120]}). Retrying in {delay:.1f}s...")
                        await asyncio.sleep(delay)

                print(f"\t\t[ANSWER] experiment={ep + 1} | money={money_value:.2f} | "
                      f"quant={quant_value:g} -> {str(response).strip()[:40]!r}")
                return {
                    "reward_value": round(float(money_value), 6),
                    "quantity": round(float(quant_value), 6),
                    "experiment": ep + 1,
                    "output": response,
                }

        def _save():
            if output_file is not None and model_name is not None:
                experiment_outcomes[model_name] = pd.DataFrame(results_list)
                with open(output_file, "wb") as f:
                    pickle.dump(experiment_outcomes, f)

        # Show exactly what is being sent (once)
        example_prompt = generate_prompt(args, money_array[0], quant_array[0])
        print(f"[DEBUG] model={model_name} | {len(quant_array)} quantities x {len(money_array)} money values "
              f"= {expected_rows} requests per experiment | semaphore={semaphore_value}")
        print("[DEBUG] Example prompt:\n" + "-" * 70 + f"\n{example_prompt}\n" + "-" * 70)

        for ep in range(args["n_experiments"]):
            ep_num = ep + 1

            if ep_num in completed_experiments:
                print(f"\tExperiment no. {ep_num}: already complete, skipping")
                continue

            print(f"\tExperiment no. {ep_num}/{args['n_experiments']}: sending {expected_rows} requests...")
            t0 = time.time()

            tasks = [
                _fetch_and_record(ep, quant_value, money_value)
                for quant_value in quant_array
                for money_value in money_array
            ]
            ep_results = await asyncio.gather(*tasks)

            n_failed = sum(1 for r in ep_results if r.get("output") is None)
            answer_counts = pd.Series(
                [str(r["output"]).strip()[:20] for r in ep_results if r.get("output") is not None]
            ).value_counts().head(5).to_dict()
            print(f"\tExperiment no. {ep_num} done in {time.time() - t0:.1f}s | "
                  f"None answers: {n_failed}/{len(ep_results)} | top answers: {answer_counts}")

            # --- stop the whole script if too many failures ---
            if n_failed > max_none_per_experiment:
                raise RuntimeError(
                    f"Aborting: experiment={ep_num} had {n_failed} None outputs "
                    f"(limit {max_none_per_experiment}). Previous experiments are already saved; this one was NOT saved."
                )

            results_list.extend(ep_results)
            _save()
            print(f"\t[SAVED] experiment {ep_num}/{args['n_experiments']} | total rows: {len(results_list)}")

    asyncio.run(_run_experiments_async())

    return pd.DataFrame(results_list)


def generate_prompt(args, money_value, quant_value):
    dollars = int(money_value)
    cents = int(round((money_value - dollars) * 100))
    if cents == 100:
        dollars += 1
        cents = 0

    template = args["prompt"]

    # .format() feeds in different values so that we have a series of prompts with different money and quantity values.
    prompt = template.format(
        reward_dollars=dollars,
        reward_cents=cents,
        quant=int(round(quant_value)),
    )
    return prompt


def initialize_models(args):
    """
    Initialize LLM models based on command line arguments.
    Args:
        args: Command line arguments containing models list and configuration
    Returns:
        dict: Dictionary of initialized model instances
    """
    args = load_config()
    # keys_dict = args.get("api_keys", {})

    with open("keys.yaml") as f:
        keys_dict = yaml.safe_load(f)

    model_configs = {
        "mistral-large-3": {
            "class": AsyncOpenRouterApi,
            "api_key": keys_dict["API_keys"]["OpenRouter"],
            "model": "mistralai/mistral-large-2512", 
            "provider_order": ["mistral/zdr"], # used to be mistral
        },
        # "mistral-large-3": {
        #     "class": AsyncMistralApi,
        #     "api_key": keys_dict["API_keys"]["Mistral"],
        #     "model": "mistral-large-2512",  
        # },
        "deepseek-v4-pro": {
            "class": AsyncOpenRouterApi,
            "api_key": keys_dict["API_keys"]["OpenRouter"],
            "model": "deepseek/deepseek-v4-pro", #0423 version
            "provider_order": ["streamlake/fp8"],
        },
        "glm-5.2": {
            "class": AsyncOpenRouterApi,
            "api_key": keys_dict["API_keys"]["OpenRouter"],
            "model": "z-ai/glm-5.2",
            "provider_order": ["deepinfra/fp4"],
        },
        "gpt-5.4-mini": {
            "class": AsyncGptApi,
            "api_key": keys_dict["API_keys"]["GptApi"],
            "model": "gpt-5.4-mini-2026-03-17",
        },
        "claude-haiku-4.5": {
            "class": AsyncClaudeApi,
            "api_key": keys_dict["API_keys"]["ClaudeApi"],
            "model": "claude-haiku-4-5-20251001",
        },
    }

    model_dict = {}
    for model_name in args["models"]:
        if model_name in model_configs:
            config = model_configs[model_name]
            model_dict[model_name] = config["class"](
                api_key=config["api_key"],
                model=config["model"],
                system_role=args["system_role"],
                temperature=args["temperature"],
                provider_order=config.get("provider_order"),
                )

        else:
            print(f"Warning: Unknown model '{model_name}' requested. Skipping.")

    return model_dict


def LR_transition_values(args, data, model_name):
    """
    Performs per-quantity logistic regression to find transition points
    for the given model_name. Rows with missing reward_value or output_binary
    are dropped. The function merges results into 'results/<prompt_type>/<experiment_name>/LR_results.pkl',
    keyed by model_name.
    Args:
        args: Command-line or config arguments (not directly used here,
              included for a consistent signature).
        data (dict): A dictionary {model_name: DataFrame} returned from your main experiment,
                     from which we extract data[model_name].
        model_name (str): Identifier for the model (e.g. "gpt4o").
    Returns:
        None. Loads and updates 'LR_results.pkl' so that LR_results[model_name] = DataFrame.
    
    Save dir:
    pickle_filename = f'{args["output_dir"]}/{args["prompt_type"]}/{args["experiment_name"]}/LR_results.pkl'

    The file directory is consisted of three layers, from original paper, it is always: 
    # results (this has been fixed)/prompt_type (such as time, pain)/experiment_name (in the begining of config_file)
    In our new project, this is same: 
    # new_results_tep2 (this will be changed to a fixed)/prompt_type (country)/experiment_name (we use test2_countrytxt, in the begining of config_file)
    
    """
    df = data[model_name].copy()
    # fill na to nan so that the following regex search won't crash due to NoneType.
    df["output"] = df["output"].fillna("")

    if args["prompt_type"] == "chinese":
        df["output_binary"] = df["output"].map(
            lambda x: (
                1
                if (
                    re.search(
                        r"^\s*是\s*(?:"
                        r"的"
                        r"|（.*?）"
                        r"|\(.*?\)"
                        r")?\s*[。.．!！?？]?\s*$",
                        str(x or "").strip(),
                    )
                    or re.search(r"\byes\b", str(x or "").strip(), flags=re.IGNORECASE)
                )
                else (
                    0
                    if (
                        re.search(r"^\s*否\s*[。.．!！?？]?\s*$", str(x or "").strip())
                        or re.search(
                            r"^\s*不(?:是)?\s*[。.．!！?？]?\s*$", str(x or "").strip()
                        )
                        or re.search(
                            r"\bno\b", str(x or "").strip(), flags=re.IGNORECASE
                        )
                    )
                    else np.nan
                )
            )
        )
    elif args["prompt_type"] == "french":
        df["output_binary"] = df["output"].map(
            lambda x: (
                1
                if re.search(r"^\s*oui\.?\s*$", str(x or ""), flags=re.IGNORECASE)
                else (
                    0
                    if re.search(r"^\s*non\.?\s*$", str(x or ""), flags=re.IGNORECASE)
                    else np.nan
                )
            )
        )
    elif args["prompt_type"] == "dutch":
        df["output_binary"] = df["output"].map(
            lambda x: (
                1
                if re.search(r"^\s*ja\.?\s*$", str(x or ""), flags=re.IGNORECASE)
                else (
                    0
                    if re.search(r"^\s*nee\.?\s*$", str(x or ""), flags=re.IGNORECASE)
                    else np.nan
                )
            )
        )
    elif args["prompt_type"] == "cot":
        df["output_binary"] = df["output"].map(
            lambda x: (
                1
                if (
                    re.search(r"answer", str(x or ""), re.IGNORECASE)
                    and re.search(r"yes\b", str(x or ""), re.IGNORECASE)
                )
                else (
                    0
                    if (
                        re.search(r"answer", str(x or ""), re.IGNORECASE)
                        and re.search(r"no\b", str(x or ""), re.IGNORECASE)
                    )
                    else np.nan
                )
            )
        )
    else:
        df["output_binary"] = df["output"].map(
            lambda x: (
                1
                if re.search(r"\byes\b", x.strip(), flags=re.IGNORECASE)
                else (
                    0
                    if re.search(r"\bno\b", x.strip(), flags=re.IGNORECASE)
                    else np.nan
                )
            )
        )

    more_than_two_words = df["output"].str.split().str.len() > 2
    if more_than_two_words.any() and args["prompt_type"] != "cot":
        print(
            "[INFO] Found outputs with more than two words at these indices for: ",
            f"**{model_name}** model in prompt_type: **{args['prompt_type']}**: ",
        )
        for idx in df.index[more_than_two_words]:
            print(f"\t-index {idx}: '{df.loc[idx, 'output']}'")
        print(
            "[INFO] Consider manual checking of the classification of aforementioned index value in 'utils.LR_transition_values' function."
        )

    mask_nan = df["reward_value"].isna() | df["output_binary"].isna() # | means OR, mask_nan will return True or False.
    df_nan = df[mask_nan] # Creates a new DataFrame containing only the rows with missing data.
    if not df_nan.empty:
        print(
            f'\nModel: **{model_name}**, prompt type: **{args["prompt_type"]}**\n',
            df_nan,
        )
        print(f"\n[INFO] Dropping {len(df_nan)} rows due to NaN.\n")
    
    df = df.dropna(subset=["reward_value", "output_binary"])

    unique_quantities = sorted(df["quantity"].unique())
    results = []

    print(f"[DEBUG][LR] model={model_name} | prompt_type={args['prompt_type']} | rows used={len(df)} | "
          f"quantities={unique_quantities} | overall yes-rate={df['output_binary'].mean():.3f}")

    for q in unique_quantities:
        sub = df[df["quantity"] == q]

        if sub.empty: # ADD safety check
            continue

        X_raw = sub[["reward_value"]].astype(float)
        if args["log_scale_money"]:
            if (X_raw <= 0).any().any():
                raise ValueError(
                    f"Found non‑positive reward_value(s) in quantity '{q}' while log_scale=True."
                )
            X = np.log10(X_raw)
        else:
            X = X_raw

        y = sub["output_binary"].astype(int)

        # Skip degenerate cases where y is constant
        if y.nunique() == 1:
            if args["log_scale_money"]:
                if y.iloc[0] == 0:
                    note_str = f'>{10**args["money_max"]}'
                else:
                    note_str = f'<{10**args["money_min"]}'
            else:
                if y.iloc[0] == 0:
                    note_str = f'>{args["money_max"]}'
                else:
                    note_str = f'<{args["money_min"]}'

            results.append((q, note_str))
            continue # The continue here is important, because you won't go to a logistic regression when y is entirely constant.

        # Fit logistic regression
        model = LogisticRegression()
        model.fit(X, y)
        beta0 = model.intercept_[0]
        beta1 = model.coef_[0][0]

        if beta1 == 0: # There is no slope. A completely flat line.
            results.append((q, "Fit slope=0?"))
            continue

        # 50% acceptance => (beta0 + beta1*x)=0 => x*=-beta0/beta1
        x_tran = -beta0 / beta1

        # if args["log_scale_money"]:
        #     x_tran = 10**x_tran

        # results.append((q, x_tran))

        # if there are scattered yes or no but not a solid and clear transition curve a logit can fit, the returned transition value could be 
        # out of the range of our given 1 dollar to 25 dollar, for example a number over 10 thousand. So we cap them:
        if args["log_scale_money"]:
            x_tran = 10**x_tran
            money_min_real = 10**args["money_min"]
            money_max_real = 10**args["money_max"]
        else:
            money_min_real = args["money_min"]
            money_max_real = args["money_max"]

        # ADD: cap out-of-range extrapolations, same logic as degenerate case
        if x_tran > money_max_real:
            results.append((q, f">{money_max_real}"))
        elif x_tran < money_min_real:
            results.append((q, f"<{money_min_real}"))
        else:
            results.append((q, x_tran))

    results_df = pd.DataFrame(results, columns=["quantity", "transition"])
    print(f"[DEBUG][LR] transition values for {model_name}:")
    print(results_df.to_string(index=False))

    # pickle_filename = f'{args["output_dir"]}/{args["prompt_type"]}/{args["experiment_name"]}/LR_results.pkl'
    # Change the filename based on log_scale_money
    if args["log_scale_money"]:
        pkl_name = "LR_results_is_log.pkl"
    else:
        pkl_name = "LR_results.pkl"

    pickle_filename = f'{args["output_dir"]}/{args["prompt_type"]}/{args["experiment_name"]}/{pkl_name}'

    try:
        with open(pickle_filename, "rb") as f:
            LR_results = pickle.load(f)
    except FileNotFoundError:
        LR_results = {}

    LR_results[model_name] = results_df
    with open(pickle_filename, "wb") as f:
        pickle.dump(LR_results, f)


def LR_values_per_quantity_table(args):
    """
    Build a trade‑off table that contains only the (prompt_type, quantity)
    pairs listed under args['trade_off_values'][experiment_name].
    - Rows  = models
    - Cols  = '<prompt_type>|<requested_quantity>'
      (shows exact value if present, or the numerically closest one)
    """
    output_dir = args["output_dir"]
    experiment_name = args["experiment_name"]
    trade_cfg = args.get("trade_off_values", {})

    if "value" in trade_cfg:
        trade_cfg = trade_cfg["value"]

    if experiment_name not in trade_cfg:
        raise KeyError(
            f"[INFO] {experiment_name!r} not found in args['trade_off_values']."
        )

    cfg_quantities = trade_cfg[experiment_name]

    all_tables = []
    for prompt_type, wanted_list in cfg_quantities.items():
        pkl_path = f"{output_dir}/{prompt_type}/{experiment_name}/LR_results.pkl"
        if not os.path.exists(pkl_path):
            print(f"[INFO] {pkl_path} not found, skipping '{prompt_type}'.")
            continue

        with open(pkl_path, "rb") as f:
            LR_results = pickle.load(f)

        # gather available quantities once
        available_q = set()
        for df in LR_results.values():
            available_q.update(df["quantity"].unique())

        numeric_available = [
            float(q)
            for q in available_q
            if isinstance(q, (int, float)) or str(q).replace(".", "", 1).isdigit()
        ]

        q_use_map = {}  # {q_req: q_use}
        for q_req in wanted_list:
            if q_req in available_q:
                q_use_map[q_req] = q_req
            elif (
                isinstance(q_req, (int, float))
                or str(q_req).replace(".", "", 1).isdigit()
            ) and numeric_available:
                q_req_num = float(q_req)
                q_match = min(numeric_available, key=lambda x: abs(x - q_req_num))
                print(
                    f"[INFO] For {prompt_type} prompt: quantity {q_req} not found, "
                    f"using closest value: {q_match} for trade-offs table."
                )
                q_use_map[q_req] = q_match
            else:
                q_use_map[q_req] = None  # nothing suitable

        rows = []
        for model_name, df in LR_results.items():
            row = {"model_name": model_name}

            for q_req in wanted_list:
                q_use = q_use_map[q_req]
                col_name = (
                    f"{prompt_type}|{q_use}"
                    if q_use is not None
                    else f"{prompt_type}|{q_req}"
                )

                if q_use is None:
                    row[col_name] = None
                    continue

                t_val = df.loc[df["quantity"] == q_use, "transition"]
                row[col_name] = None if t_val.empty else t_val.iloc[0]

            rows.append(row)

        tdf = pd.DataFrame(rows).sort_values("model_name").reset_index(drop=True)
        all_tables.append(tdf)

    if not all_tables:
        raise RuntimeError(
            "[INFO] utils.LR_values_per_quantity_table: no LR_results.pkl files processed."
        )

    tradeoff_df = (
        reduce(
            lambda left, right: pd.merge(left, right, on="model_name", how="outer"),
            all_tables,
        )
        .sort_values("model_name")
        .reset_index(drop=True)
    )

    money_min = args["money_min"]
    money_max = args["money_max"]
    log_scale = args.get("log_scale_money", False)

    if log_scale:
        if isinstance(money_min, (int, float)) and isinstance(money_max, (int, float)):
            money_min, money_max = 10**money_min, 10**money_max

    def format_value(v):
        if isinstance(v, str) or pd.isna(v):
            return v
        if not isinstance(v, (int, float)):
            return v

        if v < money_min:
            return f"<{money_min}"
        elif v > money_max:
            return f">{money_max}"
        else:
            return round(v, 2)

    num_cols = tradeoff_df.columns.drop("model_name")
    for col in num_cols:
        tradeoff_df[col] = tradeoff_df[col].apply(format_value)

    global_table_path = f"{output_dir}/{experiment_name}_tradeoff_table.pkl"
    tradeoff_df.to_pickle(global_table_path)

    print("\nTrade‑off Table (requested quantities)")
    print(tradeoff_df.to_string(index=False))
    print(f"\nSaved to: {global_table_path}")
