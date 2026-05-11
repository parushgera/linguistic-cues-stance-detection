import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# This helper function is well-designed and does not need any changes.
def _build_quartile_lookup(original_correct_path,
                           original_misclassified_path,
                           significant_features_df):
    """
    Build per-stance quartile bins + (correct, misclass) rate look-ups.
    Returns: quartile_bins_lookup, correctness_rate_lookup, misclass_rate_lookup
    """
    print("Building quartile / correctness / misclass look-ups from training data…")

    df_cor = pd.read_csv(original_correct_path)
    df_mis = pd.read_csv(original_misclassified_path)
    df_cor["label"] = "Correct"
    df_mis["label"] = "Misclassified"
    df_all = pd.concat([df_cor, df_mis], ignore_index=True)

    quartile_bins_lookup  = {}   # feature ➜ stance ➜ bins
    correctness_rate_lp   = {}   # feature ➜ stance ➜ {interval: rate}
    misclass_rate_lp      = {}

    features = significant_features_df["feature"].unique()
    st_map   = {"All Stances": None, "AGAINST": 0, "FAVOR": 1, "NONE": 2}

    for feat in features:
        quartile_bins_lookup[feat] = {}
        correctness_rate_lp [feat] = {}
        misclass_rate_lp   [feat] = {}

        for st_name, st_val in st_map.items():
            sub = df_all if st_val is None else df_all[df_all["stance"] == st_val]
            if sub.empty or feat not in sub.columns or sub[feat].nunique() < 4:
                continue

            # stance-specific bins
            try:
                _, bins = pd.qcut(sub[feat], q=4, retbins=True, duplicates="drop")
            except ValueError:
                continue

            quartile_bins_lookup[feat][st_name] = bins
            sub = sub.assign(quantile=pd.cut(sub[feat], bins=bins, include_lowest=True))

            grp = sub.groupby(["quantile", "label"], observed=False).size().unstack(fill_value=0)
            grp["Total"]            = grp.sum(axis=1)
            grp["Correct Rate"]     = grp["Correct"]      / grp["Total"]
            grp["Misclassified Rate"] = grp["Misclassified"] / grp["Total"]

            correctness_rate_lp[feat][st_name] = grp["Correct Rate"].to_dict()
            misclass_rate_lp  [feat][st_name] = grp["Misclassified Rate"].to_dict()

    print("… look-ups ready.\n")
   
    return quartile_bins_lookup, correctness_rate_lp, misclass_rate_lp


# MODIFIED: Added 'output_dir' parameter to control where results are saved.
def evaluate_stylistic_confidence(evaluation_folder,
                                  significant_features_path,
                                  original_correct_path,
                                  original_misclassified_path,
                                  output_dir): # <-- NEW PARAMETER
    """
    Compute per-sample stylistic confidence scores and save results to a specific directory.
    """
    # 1. Load evaluation data
    print("[1] Loading evaluation set …")
    eval_df = pd.read_csv(os.path.join(evaluation_folder,
                                       "wtwt_test_processed.csv"))
    corr_idx = np.load(os.path.join(evaluation_folder,
                                    "correctly_classified_indices.npy"))
    mis_idx  = np.load(os.path.join(evaluation_folder,
                                    "misclassified_indices.npy"))

    eval_df["actual_outcome"]        = "Correct"
    eval_df.loc[mis_idx, "actual_outcome"] = "Misclassified"
    eval_df["actual_outcome_binary"] = eval_df["actual_outcome"].map(
        {"Correct": 0, "Misclassified": 1}).astype(int)

    # 2. Significant features
    print("[2] Loading significant features from:", significant_features_path)
    sig_df = pd.read_csv(significant_features_path)
    
    # 3. Build look-ups from training data
    bins_lp, corr_lp, mis_lp = _build_quartile_lookup(
        original_correct_path, original_misclassified_path, sig_df)

    # 4. Score each sample
    print("[3] Calculating scores for each sample…")
    # ... (scoring logic is perfect, omitting for brevity) ...
    print("[3] Calculating scores …")
    st_inv = {0: "AGAINST", 1: "FAVOR", 2: "NONE"}
    results = []

    for idx, row in eval_df.iterrows():
        st_val  = row.get("stance")
        st_name = st_inv.get(st_val)           # None if stance missing
        if st_name is None:
            continue

        feat_subset = sig_df[sig_df["stance"] == st_name]["feature"].unique()

        corr_rates, mis_rates = [], []

        for feat in feat_subset:
            if pd.isna(row.get(feat)):
                continue

            # choose most specific bins available
#             bins = bins_lp.get(feat, {}).get(st_name)
#             if bins is None:                       # first try failed
#                 bins = bins_lp.get(feat, {}).get("All Stances")
#             if bins is None:                       # still nothing ⇒ skip
#                 continue

#             interval = pd.cut([row[feat]], bins=bins, include_lowest=True)[0]
#             corr_rate = corr_lp.get(feat, {}).get(st_name, {}).get(interval) \
#                         or corr_lp.get(feat, {}).get("All Stances", {}).get(interval)
#             mis_rate  = mis_lp .get(feat, {}).get(st_name, {}).get(interval) \
#                         or mis_lp .get(feat, {}).get("All Stances", {}).get(interval)

#             if corr_rate is not None: corr_rates.append(corr_rate)
#             if mis_rate  is not None: mis_rates .append(mis_rate)
            
            bins = bins_lp.get(feat, {}).get(st_name)
            if bins is None: 
                continue # no bins for this stance ⇒ skip feature
            interval   = pd.cut([row[feat]], bins=bins, include_lowest=True)[0]
            corr_rate  = corr_lp.get(feat, {}).get(st_name, {}).get(interval)
            mis_rate   = mis_lp .get(feat, {}).get(st_name, {}).get(interval)
            if corr_rate is not None: corr_rates.append(corr_rate)
            if mis_rate  is not None: mis_rates .append(mis_rate)
            

        score_corr = np.mean(corr_rates) if corr_rates else np.nan
        score_mis  = np.mean(mis_rates)  if mis_rates  else np.nan

        results.append({
            "index": idx,
            "score_correct"      : score_corr,
            "score_misclassified": score_mis,
            "actual_outcome"     : row["actual_outcome"],
            "actual_outcome_binary": row["actual_outcome_binary"]
        })

    results_df = pd.DataFrame(results).dropna()
    print("… scoring done.\n")


    # 5. Quick stats
    print("[4] Aggregate statistics")
    print(results_df.groupby("actual_outcome")[["score_correct", "score_misclassified"]].mean().round(3))


    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    sns.boxplot(x="actual_outcome", y="score_correct",
                data=results_df, order=["Correct","Misclassified"])
    plt.title("Expected CORRECTNESS")

    plt.subplot(1,2,2)
    sns.boxplot(x="actual_outcome", y="score_misclassified",
                data=results_df, order=["Correct","Misclassified"])
    plt.title("Expected MISCLASSIFICATION")

    plt.tight_layout()

    
    # MODIFIED: Use the new output_dir for saving plots and results.
    out_plot_boxplot = os.path.join(output_dir, "confidence_dual_boxplot.png")
    plt.savefig(out_plot_boxplot, dpi=300)
    print(f"Box plots saved to ➜ {out_plot_boxplot}")
    plt.close() # Close the figure to free up memory

    # 7. Save the detailed results CSV to the output directory
    out_csv = os.path.join(output_dir, "evaluation_results.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"Evaluation results CSV saved to ➜ {out_csv}")
    
    # Return the results for any further interactive analysis
    return results_df


if __name__ == '__main__':
    sns.set_theme(style="whitegrid")

    # --- 1. Define Your Experiments ---
    # List of configurations for each experimental run.
    experiments_to_run = [
        {'coef': 'positive', 'sig': 'yes'},
        {'coef': 'positive', 'sig': 'no'},
        {'coef': 'negative', 'sig': 'yes'},
        {'coef': 'negative', 'sig': 'no'}
    ]
    coef_thresholds = np.round(np.arange(0.10, 1.00, 0.10), 2).tolist()
    
    for threshold in coef_thresholds:
        BASE_PLOTS_DIR = 'paper_plots_v3' # Changed to v3 as per your description
        THRESHOLD_DIR = f"{threshold}_threshold_experiment"
        EVALUATION_DATA_FOLDER = 'evaluation_data' # Folder with wtwt_test_processed.csv etc.
        ORIGINAL_CORRECT_DATA_PATH = '/home/p/parush/style_markers/classifications/whole/correctly_classified_examples_processed.csv'
        ORIGINAL_MISCLASSIFIED_DATA_PATH = '/home/p/parush/style_markers/classifications/whole/misclassified_examples_processed.csv'

        # Assuming dataset and source are constant for these runs
        dataset = 'whole'
        source = 'whole'

        # --- 3. Loop Through and Run All Experiments ---
        for i, experiment_config in enumerate(experiments_to_run):
            coef_direction = experiment_config['coef']
            significance_filter = experiment_config['sig']

            print(f"\n{'='*60}")
            print(f"  RUNNING EVALUATION FOR EXPERIMENT {i+1}/{len(experiments_to_run)}")
            print(f"  Coefficient Direction: '{coef_direction}', Significance Filter: '{significance_filter}'")
            print(f"{'='*60}\n")

            # Dynamically construct the paths for this specific experiment
            experiment_name = f"{source}_{dataset}_coef-{coef_direction}_sig-{significance_filter}"
            experiment_folder = os.path.join(BASE_PLOTS_DIR,THRESHOLD_DIR, experiment_name)

            # This is the "ruleset" file created by your analysis script
            ruleset_path = os.path.join(experiment_folder, "filtered_features_summary.csv")

            # This is the new subfolder where results will be saved
            evaluation_output_dir = os.path.join(experiment_folder, "evaluation_results")
            os.makedirs(evaluation_output_dir, exist_ok=True)

            if not os.path.exists(ruleset_path):
                print(f"  [ERROR] Ruleset file not found, skipping: {ruleset_path}")
                continue

            # Call the main evaluation function with the correct paths for this run
            evaluation_results = evaluate_stylistic_confidence(
                evaluation_folder=EVALUATION_DATA_FOLDER,
                significant_features_path=ruleset_path,
                original_correct_path=ORIGINAL_CORRECT_DATA_PATH,
                original_misclassified_path=ORIGINAL_MISCLASSIFIED_DATA_PATH,
                output_dir=evaluation_output_dir # Pass the specific output directory
            )

            # --- Plotting section (now inside the loop) ---
            print("\n[5] Generating additional plots...")
            for score_col in ["score_correct", "score_misclassified"]:
                plt.figure(figsize=(7, 4))
                sns.histplot(data=evaluation_results, x=score_col, hue="actual_outcome",
                             element="step", stat="density", common_norm=False, bins=20)
                plt.xlabel(f"Predicted {score_col.split('_')[1]} probability")
                plt.title("Histogram of scores by true outcome")
                plt.tight_layout()

                hist_plot_path = os.path.join(evaluation_output_dir, f"hist_{score_col}.png")
                plt.savefig(hist_plot_path, dpi=300)
                print(f"Histogram saved to ➜ {hist_plot_path}")
                plt.close() # Close each plot

        print("\n\nAll evaluations complete.")