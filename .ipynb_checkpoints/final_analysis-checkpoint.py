significance_df = pd.read_csv("all_feature_significance_summary.csv")

dataset ='whole'    # ['covid', 'pstance', 'semeval', 'whole']
source = 'whole'   # ['overall', 'target', 'whole']]

if source == 'whole' and dataset == 'whole':
    print(f"Working on Source: {source} and Dataset: {dataset}")
    correct_path = f"/home/p/parush/style_markers/classifications/{dataset}/correctly_classified_examples_processed.csv"
    misclassified_path = f"/home/p/parush/style_markers/classifications/{dataset}/misclassified_examples_processed.csv"
    print(f"correct_path: {correct_path}")
    print(f"misclassified_path: {misclassified_path}")
else:
    print(f"Working on Source: {source} and Dataset: {dataset}")
    correct_path = f"/home/p/parush/style_markers/classifications/{dataset}/{source}/correctly_classified_examples_processed.csv"
    misclassified_path = f"/home/p/parush/style_markers/classifications/{dataset}/{source}/misclassified_examples_processed.csv"
    print(f"correct_path: {correct_path}")
    print(f"misclassified_path: {correct_path}")
    
experiments_to_run = [
        {'coef': 'positive', 'sig': 'yes'},
        {'coef': 'positive', 'sig': 'no'},
        {'coef': 'negative', 'sig': 'yes'},
        {'coef': 'negative', 'sig': 'no'}
    ]

coef_thresholds = np.round(np.arange(0.10, 1.00, 0.10), 2).tolist()

for thr in coef_thresholds:                                   # ‹thr› = current threshold
    print(f"\n{'#' * 70}")
    print(f"★  STARTING THRESHOLD RUN  →  coef_threshold = {thr:.2f}")
    print(f"{'#' * 70}")

    for run_idx, cfg in enumerate(experiments_to_run, start=1):
        coef_dir = cfg["coef"]          # 'positive' | 'negative' | 'all'
        sig_filt = cfg["sig"]           # 'yes' | 'no' | 'all'

        print(f"\n{'-'*50}")
        print(f"  ▶ Experiment {run_idx:02d}/{len(experiments_to_run)}")
        print(f"     • Coefficient direction : {coef_dir}")
        print(f"     • Significance filter   : {sig_filt}")
        print(f"     • Threshold (|β|)       : {thr:.2f}")
        print(f"{'-'*50}")

        _ = analyze_lr_significance_quartiles_with_filter_and_summary(
                correct_path      = correct_path,
                misclassified_path= misclassified_path,
                significance_df   = significance_df,
                dataset           = dataset,
                source            = source,
                coef_direction    = coef_dir,
                significance_filter = sig_filt,
                coef_threshold    = thr,          # ← current threshold
                quantiles         = 4
        )


# coef_direction = 'all'
# significance_filter = 'yes'
# print(f"\n{'='*60}")
# print(f"  RUNNING EVALUATION FOR EXPERIMENT ")
# print(f"  Coefficient Direction: '{coef_direction}', Significance Filter: '{significance_filter}'")
# print(f"{'='*60}\n")
# lr_results = analyze_lr_significance_quartiles_with_filter_and_summary(
#     correct_path=correct_path,
#     misclassified_path=misclassified_path,
#     significance_df=significance_df,
#     coef_threshold=0.1, 
#     coef_direction=coef_direction,
#     significance_filter=significance_filter,# load this from your significance summary CSV
#     dataset=dataset,
#     source=source,  # or "overall" or "whole"
#     quantiles=4
# )
