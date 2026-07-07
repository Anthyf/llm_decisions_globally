import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample


def bootstrap_LR_analysis(args, data, model_name, n_bootstrap=20, save_plots=True):
    """
    Performs bootstrap resampling on the data and fits logistic regression models
    to each bootstrap sample, calculating transition values and confidence intervals.

    Args:
        args: Command-line or config arguments
        data (dict): A dictionary {model_name: DataFrame} from the main experiment
        model_name (str): Identifier for the model (e.g. "gpt4o")
        n_bootstrap (int): Number of bootstrap samples to generate
        save_plots (bool): Whether to save plots to disk

    Returns:
        dict: Dictionary with transition values and confidence intervals for each quantity
    """
    # Ensure output directory exists if we're saving plots
    if save_plots and isinstance(args, dict):
        os.makedirs(
            f'{args["output_dir"]}/{args["prompt_type"]}/{args["experiment_name"]}',
            exist_ok=True,
        )

    df = data[model_name].copy()

    # Drop rows with missing values
    df = df.dropna(subset=["reward_value", "output_binary"])

    # Get unique quantities
    unique_quantities = sorted(df["quantity"].unique())
    bootstrap_results = {}

    # For each quantity, perform bootstrap analysis
    for q in unique_quantities:
        sub_df = df[df["quantity"] == q]

        # Skip quantities with too few data points
        if len(sub_df) < 5:
            print(
                f"Skipping quantity '{q}' due to insufficient data points ({len(sub_df)} available)"
            )
            continue

        # Skip degenerate cases where all responses are the same
        if sub_df["output_binary"].nunique() == 1:
            print(f"Skipping quantity '{q}' because all responses are the same")
            continue

        # Store all bootstrap transitions for this quantity
        transitions = []

        # Create bootstrap samples and fit models
        seed = 42
        np.random.seed(seed)
        for i in range(n_bootstrap):
            # Create bootstrap sample with replacement
            bootstrap_sample = resample(sub_df, replace=True, n_samples=len(sub_df))

            # Skip if the bootstrap sample is degenerate (all yes or all no)
            if bootstrap_sample["output_binary"].nunique() == 1:
                continue

            # Prepare data for logistic regression
            X_raw = bootstrap_sample[["reward_value"]].astype(float)

            # Apply log scaling if specified
            if args.get("log_scale_money", False) or (
                isinstance(args, dict) and args.get("log_scale_money", False)
            ):
                if (X_raw <= 0).any().any():
                    continue  # Skip bootstrap samples with non-positive values in log scale
                X = np.log10(X_raw)
            else:
                X = X_raw

            y = bootstrap_sample["output_binary"].astype(int)

            # Fit logistic regression
            try:
                model = LogisticRegression(
                    max_iter=1000 # by default it will be 100 if we don't set it.
                )  # Increase max_iter to ensure convergence
                model.fit(X, y)

                beta0 = model.intercept_[0]
                beta1 = model.coef_[0][0]

                # Skip if the slope is too close to zero (flat line)
                if abs(beta1) < 1e-6:
                    continue

                # Calculate transition point (-beta0/beta1)
                x_star = -beta0 / beta1

                # Convert from log scale if needed
                if args.get("log_scale_money", False) or (
                    isinstance(args, dict) and args.get("log_scale_money", False)
                ):
                    x_star = 10**x_star

                transitions.append(x_star)
            except Exception as e:
                print(f"Error fitting bootstrap sample for quantity '{q}': {e}")
                continue

        # Calculate statistics from bootstrap samples
        if transitions:
            transitions = np.array(transitions)
            mean_transition = np.mean(transitions)
            median_transition = np.median(transitions)
            ci_low = np.percentile(transitions, 2.5)  # 2.5th percentile for 95% CI
            ci_high = np.percentile(transitions, 97.5)  # 97.5th percentile for 95% CI

            bootstrap_results[q] = {
                "transitions": transitions,
                "mean": mean_transition,
                "median": median_transition,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }

            # Plot if requested
            if save_plots:
                plt.figure(figsize=(10, 6))

                # Plot histogram of bootstrap transitions
                plt.hist(transitions, bins=20, alpha=0.7, color="skyblue")

                # Add vertical lines for mean, median, and confidence interval
                plt.axvline(
                    mean_transition,
                    color="red",
                    linestyle="-",
                    label=f"Mean: ${mean_transition:.2f}",
                )
                plt.axvline(
                    median_transition,
                    color="green",
                    linestyle="--",
                    label=f"Median: ${median_transition:.2f}",
                )
                plt.axvline(
                    ci_low,
                    color="purple",
                    linestyle=":",
                    label=f"95% CI: ${ci_low:.2f}",
                )
                plt.axvline(
                    ci_high, color="purple", linestyle=":", label=f"to ${ci_high:.2f}"
                )

                # Add labels and title
                plt.xlabel("Transition Value")
                plt.ylabel("Frequency")
                plt.title(f"Bootstrap Distribution for {q} (N={len(transitions)})")
                plt.legend()

                # Adjust x-axis for readable values
                if not args.get("log_scale_money", False):
                    plt.ticklabel_format(style="plain", axis="x")

                # Save figure
                if isinstance(args, dict):
                    plot_path = f'{args["output_dir"]}/{args["prompt_type"]}/{args["experiment_name"]}/bootstrap_{model_name}_{q}_{n_bootstrap}.png'
                    plt.tight_layout()
                    plt.savefig(plot_path, dpi=300)
                    print(f"Saved bootstrap plot for '{q}' to {plot_path}")
                else:
                    plt.tight_layout()
                    plt.show()
                plt.close()
        else:
            print(f"No valid bootstrap samples for quantity '{q}'")

    # Create a summary plot with all quantities
    if save_plots and bootstrap_results:
        plt.figure(figsize=(12, 8))

        quantities = list(bootstrap_results.keys())
        means = [bootstrap_results[q]["mean"] for q in quantities]
        ci_lows = [bootstrap_results[q]["ci_low"] for q in quantities]
        ci_highs = [bootstrap_results[q]["ci_high"] for q in quantities]

        # Error bars represent 95% confidence intervals
        y_pos = np.arange(len(quantities))
        error = [
            np.array(means) - np.array(ci_lows),
            np.array(ci_highs) - np.array(means),
        ]

        plt.errorbar(means, y_pos, xerr=error, fmt="o", capsize=5, color="blue")

        # Add quantity labels
        plt.yticks(y_pos, quantities)

        # Add labels and title
        plt.xlabel("Transition Value ($)")
        plt.title(f"Transition Values with 95% CI - {model_name}")

        # Adjust x-axis for readable values
        plt.ticklabel_format(style="plain", axis="x")

        # Add grid for better readability
        plt.grid(axis="x", linestyle="--", alpha=0.7)

        # Save figure
        if isinstance(args, dict):
            summary_path = f'{args["output_dir"]}/{args["prompt_type"]}/{args["experiment_name"]}/bootstrap_summary_{model_name}_{n_bootstrap}.png'
            plt.tight_layout()
            plt.savefig(summary_path, dpi=300)
            print(f"Saved bootstrap summary plot to {summary_path}")
        else:
            plt.tight_layout()
            plt.show()
        plt.close()

    # Save bootstrap results to file
    if isinstance(args, dict):
        results_path = f'{args["output_dir"]}/{args["prompt_type"]}/{args["experiment_name"]}/bootstrap_results_{model_name}_{n_bootstrap}.pkl'
        with open(results_path, "wb") as f:
            pickle.dump(bootstrap_results, f)
        print(f"Saved bootstrap results to {results_path}")

    return bootstrap_results


# Calling:

# n_bootstrap = 30
# bootstrap_results = {}
# bootstrap_results[f"bs_{n_bootstrap}"] = bootstrap_LR_analysis(
#     args, data, "claude3_5", n_bootstrap=n_bootstrap
# )

# print(
#     f"\nMean TP for BS n_{n_bootstrap}: ",
#     round(
#         bootstrap_results[f"bs_{n_bootstrap}"][np.float64(5.0)]["mean"], # in 'np.float64(5.0)', 5.0 is the quantity value for which we want to get the mean transition point after bootstrapping.
#         4,
#     ),
#     f"\nMedian TP for BS n_{n_bootstrap}: ",
#     round(
#         bootstrap_results[f"bs_{n_bootstrap}"][np.float64(5.0)][ # same here
#             "median"
#         ],
#         4,
#     ),
# )