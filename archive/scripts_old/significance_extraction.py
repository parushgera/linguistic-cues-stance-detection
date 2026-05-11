from pathlib import Path
import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu
import numpy as np
from itertools import product
from scipy.stats import ttest_ind, mannwhitneyu, shapiro

def cohens_d(x, y):
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx - 1)*np.var(x, ddof=1) + (ny - 1)*np.var(y, ddof=1)) / (nx + ny - 2))
    return (np.mean(x) - np.mean(y)) / pooled_std if pooled_std else 0.0

def cliffs_delta(x, y):
    n_x, n_y = len(x), len(y)
    more = sum(1 for a, b in product(x, y) if a > b)
    less = sum(1 for a, b in product(x, y) if a < b)
    return round((more - less) / (n_x * n_y), 4)

def is_normal(data, alpha=0.05):
    if len(data) < 3:
        return False
    stat, p = shapiro(data)
    return p > alpha

def interpret_effect_size(cohen_d, cliffs_d):
    d_strength = ("small" if abs(cohen_d) < 0.3 else
                  "medium" if abs(cohen_d) < 0.7 else
                  "large")

    cd_strength = ("negligible" if abs(cliffs_d) < 0.147 else
                   "small" if abs(cliffs_d) < 0.33 else
                   "medium" if abs(cliffs_d) < 0.474 else
                   "large")

    return d_strength, cd_strength
def load_classification_data(base_dir, dataset, target=None, stance=None, source="target"):
    """
    Load processed correctly and misclassified examples with features.

    Parameters:
    - base_dir: Root folder (e.g., "classifications")
    - dataset: Dataset name ("covid", "semeval", "pstance", "whole")
    - target: Target name (e.g., "Fauci") or None
    - stance: Filter stance ("AGAINST", "FAVOR", "NONE") or None
    - source: "target", "overall", or "whole"

    Returns:
    - Concatenated DataFrame of correctly and misclassified examples, with label column
    """
    dataset_path = Path(base_dir) / dataset

    if source == "whole":
        if dataset != 'whole':
            raise ValueError("The dataset must be 'Whole'")
        if target:
            raise ValueError("Remove target, we don' need it'")
            
        # Load from dataset/ (which is 'whole' in this case)
        correct_file = dataset_path / "correctly_classified_examples_processed.csv"
        misclassified_file = dataset_path / "misclassified_examples_processed.csv"

        correct_df = pd.read_csv(correct_file)
        misclassified_df = pd.read_csv(misclassified_file)


    elif source == "overall":
        if dataset == 'whole':
            raise ValueError("You must specify a valid dataset.")
        if target:
            raise ValueError("Remove target, we don' need it'")
            

        overall_path = dataset_path / "overall"
        correct_file = overall_path / "correctly_classified_examples_processed.csv"
        misclassified_file = overall_path / "misclassified_examples_processed.csv"

        correct_df = pd.read_csv(correct_file)
        misclassified_df = pd.read_csv(misclassified_file)


    elif source == "target":
        if target is None:
            raise ValueError("You must specify a target when source='target'.")
        if dataset == 'whole':
            raise ValueError("You must specify a valid dataset.")

        target_path = dataset_path / target
        correct_file = target_path / "correctly_classified_examples_processed.csv"
        misclassified_file = target_path / "misclassified_examples_processed.csv"

        correct_df = pd.read_csv(correct_file)
        misclassified_df = pd.read_csv(misclassified_file)

    else:
        raise ValueError(f"Invalid source: {source}. Must be one of 'target', 'overall', 'whole'.")

    # Filter stance
    stance_map = {"AGAINST": 0, "FAVOR": 1, "NONE": 2}

    if stance:
        stance_code = stance_map.get(stance)
        if stance_code is None:
            raise ValueError(f"Invalid stance provided: {stance}")
        correct_df = correct_df[correct_df["stance"] == stance_code]
        misclassified_df = misclassified_df[misclassified_df["stance"] == stance_code]

    correct_df["label"] = "Correct"
    misclassified_df["label"] = "Misclassified"

    return pd.concat([correct_df, misclassified_df], ignore_index=True)


def scan_all_features(
    base_dir,
    features,
    datasets,
    stances=["AGAINST", "FAVOR", "NONE"],
    sources=["target", "overall", "whole"]
):
    results = []
    dataset_stance_map = {
    "covid": ["AGAINST", "FAVOR", "NONE"],
    "semeval": ["AGAINST", "FAVOR", "NONE"],
    "pstance": ["AGAINST", "FAVOR"],  # No NONE class
    "whole": ["AGAINST", "FAVOR", "NONE"]
    }

    for dataset in datasets:
        stances = dataset_stance_map.get(dataset, ["AGAINST", "FAVOR", "NONE"])
        dataset_path = Path(base_dir) / dataset
        if not dataset_path.exists():
            print("dataset not found")
            continue

        # Handle 'whole' separately (only dataset='whole' and source='whole')
        if dataset == 'whole' and "whole" in sources:
            for stance in stances:
                try:
                    df = load_classification_data(base_dir, dataset='whole', stance=stance, source="whole")
                    print(base_dir)
                    print('Loaded', len(df))
                    for feature in features:
                        if feature not in df.columns:
                            continue

                        correct = df[df["label"] == "Correct"][feature].dropna()
                        misclassified = df[df["label"] == "Misclassified"][feature].dropna()

                        if len(correct) < 5 or len(misclassified) < 5:
                            continue

                        t_stat, t_p = ttest_ind(correct, misclassified, equal_var=False)
                        u_stat, u_p = mannwhitneyu(correct, misclassified, alternative='two-sided')

                        d = cohens_d(correct, misclassified)
                        cd = cliffs_delta(correct, misclassified)
                        d_strength, cd_strength = interpret_effect_size(d, cd)

                        distributions_normal = is_normal(correct) and is_normal(misclassified)
                        preferred_effect = "Cohen's d" if distributions_normal else "Cliff's delta"
                        preferred_strength = d_strength if distributions_normal else cd_strength

                        results.append({
                            "feature": feature,
                            "dataset": 'whole',
                            "target": "ALL",
                            "stance": stance,
                            "source": "whole",
                            "t_pvalue": round(t_p, 4),
                            "u_pvalue": round(u_p, 4),
                            "cohens_d": round(d, 4),
                            "cliffs_delta": cd,
                            "cohen_strength": d_strength,
                            "cliff_strength": cd_strength,
                            "distributions_normal": distributions_normal,
                            "preferred_effect": preferred_effect,
                            "preferred_strength": preferred_strength,
                            "significant": "yes" if u_p < 0.05 or t_p < 0.05 else "no"
                        })
                except Exception as e:
                    print(f"[SKIP] whole/{stance}: {e}")
                    continue

        else:  # datasets covid, semeval, pstance
            # Handle source='overall' (once per dataset)
            if "overall" in sources:
                for stance in stances:
                    try:
                        df = load_classification_data(base_dir, dataset, stance=stance, source="overall")
                        for feature in features:
                            if feature not in df.columns:
                                continue

                            correct = df[df["label"] == "Correct"][feature].dropna()
                            misclassified = df[df["label"] == "Misclassified"][feature].dropna()

                            if len(correct) < 5 or len(misclassified) < 5:
                                continue

                            t_stat, t_p = ttest_ind(correct, misclassified, equal_var=False)
                            u_stat, u_p = mannwhitneyu(correct, misclassified, alternative='two-sided')

                            d = cohens_d(correct, misclassified)
                            cd = cliffs_delta(correct, misclassified)
                            d_strength, cd_strength = interpret_effect_size(d, cd)

                            distributions_normal = is_normal(correct) and is_normal(misclassified)
                            preferred_effect = "Cohen's d" if distributions_normal else "Cliff's delta"
                            preferred_strength = d_strength if distributions_normal else cd_strength

                            results.append({
                                "feature": feature,
                                "dataset": dataset,
                                "target": "ALL",
                                "stance": stance,
                                "source": "overall",
                                "t_pvalue": round(t_p, 4),
                                "u_pvalue": round(u_p, 4),
                                "cohens_d": round(d, 4),
                                "cliffs_delta": cd,
                                "cohen_strength": d_strength,
                                "cliff_strength": cd_strength,
                                "distributions_normal": distributions_normal,
                                "preferred_effect": preferred_effect,
                                "preferred_strength": preferred_strength,
                                "significant": "yes" if u_p < 0.05 or t_p < 0.05 else "no"
                            })
                    except Exception as e:
                        print(f"[SKIP] {dataset}/overall/{stance}: {e}")
                        continue

            # Handle source='target' (per target)
            if "target" in sources:
                targets = [f.name for f in dataset_path.iterdir() if f.is_dir() and f.name != "overall"]
                for target in targets:
                    for stance in stances:
                        try:
                            df = load_classification_data(base_dir, dataset, target=target, stance=stance, source="target")
                            for feature in features:
                                if feature not in df.columns:
                                    continue

                                correct = df[df["label"] == "Correct"][feature].dropna()
                                misclassified = df[df["label"] == "Misclassified"][feature].dropna()

                                if len(correct) < 5 or len(misclassified) < 5:
                                    continue

                                t_stat, t_p = ttest_ind(correct, misclassified, equal_var=False)
                                u_stat, u_p = mannwhitneyu(correct, misclassified, alternative='two-sided')

                                d = cohens_d(correct, misclassified)
                                cd = cliffs_delta(correct, misclassified)
                                d_strength, cd_strength = interpret_effect_size(d, cd)

                                distributions_normal = is_normal(correct) and is_normal(misclassified)
                                preferred_effect = "Cohen's d" if distributions_normal else "Cliff's delta"
                                preferred_strength = d_strength if distributions_normal else cd_strength

                                results.append({
                                    "feature": feature,
                                    "dataset": dataset,
                                    "target": target,
                                    "stance": stance,
                                    "source": "target",
                                    "t_pvalue": round(t_p, 4),
                                    "u_pvalue": round(u_p, 4),
                                    "cohens_d": round(d, 4),
                                    "cliffs_delta": cd,
                                    "cohen_strength": d_strength,
                                    "cliff_strength": cd_strength,
                                    "distributions_normal": distributions_normal,
                                    "preferred_effect": preferred_effect,
                                    "preferred_strength": preferred_strength,
                                    "significant": "yes" if u_p < 0.05 or t_p < 0.05 else "no"
                                })
                        except Exception as e:
                            print(f"[SKIP] {dataset}/{target}/{stance}: {e}")
                            continue

    return pd.DataFrame(results)

if __name__ == '__main__'

STYLE_FEATURES =  [
    "word_count", "sentence_count", "avg_word_length", "std_word_length", "type_token_ratio", "hapax_legomena",
    "stopword_ratio", "punctuation_density", "noun_ratio", "verb_ratio", "adj_ratio", "adv_ratio", "pronoun_ratio",
    "avg_dependency_depth", "function_word_ratio", "punctuation_usage", "readability_score",
    "sentence_length_variation", "subordinate_clause_ratio", "sentiment_polarity", "sentiment_subjectivity",
    "anger", "joy", "fear", "sadness", "disgust", "surprise",
    "certainty_adverbs_count", "certainty_verbs_count", "certainty_adjectives_count",
    "doubt_adverbs_count", "doubt_verbs_count", "doubt_adjectives_count",
    "hedges_score", "hedge_C_count", "hedge_D_count", "hedge_E_count",
    "hedge_I_count", "hedge_N_count",
    "C_ratio", "D_ratio", "E_ratio", "I_ratio", "N_ratio"
]

df_all_results = scan_all_features(
    base_dir="classifications",
    features=STYLE_FEATURES,
    datasets=["whole", "covid", "pstance", "semeval"] 
)

sig_only = df_all_results[df_all_results["significant"] == "yes"]

sig_only.to_csv("all_feature_signal_only.csv", index=False)