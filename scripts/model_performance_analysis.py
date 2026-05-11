#!/usr/bin/env python3
"""
Model Performance Analysis: F1 Score Evaluation Across Stances and Models
=========================================================================

This script trains Logistic Regression and XGBoost models across 13 balanced iterations
for each stance label and calculates mean F1 scores with standard deviations.

Focus: Pure performance evaluation without SHAP analysis for faster execution.
"""

import os
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

def load_and_prepare_data(base_dir):
    """Load data and return separate correct and misclassified dataframes"""
    paths = ['wtwt_test_processed.csv', 'except_wtwt_test_processed_mapped_data.csv']
    indices = ['wtwt_correctly_classified_indices.npy', 'wtwt_misclassified_indices.npy',
               'except_wtwt_correctly_classified_indices.npy', 'except_wtwt_misclassified_indices.npy']

    df_wtwt = pd.read_csv(os.path.join(base_dir, paths[0]))
    df_except = pd.read_csv(os.path.join(base_dir, paths[1]))

    wtwt_correct = df_wtwt.iloc[np.load(os.path.join(base_dir, indices[0]))]
    wtwt_mis = df_wtwt.iloc[np.load(os.path.join(base_dir, indices[1]))]
    except_correct = df_except.iloc[np.load(os.path.join(base_dir, indices[2]))]
    except_mis = df_except.iloc[np.load(os.path.join(base_dir, indices[3]))]

    df_correct = pd.concat([wtwt_correct, except_correct]).assign(label=1)
    df_mis = pd.concat([wtwt_mis, except_mis]).assign(label=0)
    
    print(f"Correctly classified: {len(df_correct)}, Misclassified: {len(df_mis)}")
    
    return df_correct.reset_index(drop=True), df_mis.reset_index(drop=True)

def create_balanced_datasets_all_data(df_correct, df_mis):
    """Create 13 balanced datasets using ALL correctly classified examples"""
    balanced_datasets = []
    misclassified_stance_counts = df_mis['stance'].value_counts()
    
    # Calculate iterations
    full_iterations = len(df_correct) // len(df_mis)
    remaining_examples = len(df_correct) % len(df_mis)
    
    print(f"Creating {full_iterations} full iterations + 1 oversampled iteration")
    print(f"Full iterations use {full_iterations * len(df_mis)} examples")
    print(f"Final iteration uses remaining {remaining_examples} examples + oversampling")
    
    # Create full iterations
    for iteration in range(full_iterations):
        df_correct_sampled_parts = []
        
        for stance in misclassified_stance_counts.index:
            stance_correct = df_correct[df_correct['stance'] == stance]
            stance_misclassified_count = misclassified_stance_counts[stance]
            
            if len(stance_correct) > 0:
                total_available = len(stance_correct)
                samples_per_iteration = total_available // full_iterations
                
                start_idx = iteration * samples_per_iteration
                end_idx = (iteration + 1) * samples_per_iteration
                
                sampled_correct = stance_correct.iloc[start_idx:end_idx]
                df_correct_sampled_parts.append(sampled_correct)
        
        if df_correct_sampled_parts:
            df_correct_sampled = pd.concat(df_correct_sampled_parts)
            df_balanced = pd.concat([df_correct_sampled, df_mis]).reset_index(drop=True)
            balanced_datasets.append(df_balanced)
    
    # Create 13th iteration with oversampling
    if remaining_examples > 0:
        df_correct_13th_parts = []
        
        for stance in misclassified_stance_counts.index:
            stance_correct = df_correct[df_correct['stance'] == stance]
            stance_misclassified_count = misclassified_stance_counts[stance]
            
            if len(stance_correct) > 0:
                total_available = len(stance_correct)
                samples_per_iteration = total_available // full_iterations
                
                # Get remaining examples for this stance
                used_for_stance = full_iterations * samples_per_iteration
                remaining_for_stance = total_available - used_for_stance
                
                if remaining_for_stance > 0:
                    remaining_correct = stance_correct.iloc[used_for_stance:]
                    
                    # Calculate how many we need to reach the target
                    needed_samples = stance_misclassified_count
                    
                    if len(remaining_correct) >= needed_samples:
                        sampled_correct = remaining_correct.iloc[:needed_samples]
                    else:
                        # Oversample by randomly sampling from all used examples for this stance
                        all_used_for_stance = stance_correct.iloc[:used_for_stance]
                        additional_needed = needed_samples - len(remaining_correct)
                        
                        if len(all_used_for_stance) > 0:
                            additional_samples = all_used_for_stance.sample(
                                n=min(additional_needed, len(all_used_for_stance)), 
                                random_state=42
                            )
                            sampled_correct = pd.concat([remaining_correct, additional_samples])
                        else:
                            sampled_correct = remaining_correct
                    
                    df_correct_13th_parts.append(sampled_correct)
        
        if df_correct_13th_parts:
            df_correct_13th = pd.concat(df_correct_13th_parts)
            df_balanced_13th = pd.concat([df_correct_13th, df_mis]).reset_index(drop=True)
            balanced_datasets.append(df_balanced_13th)
    
    print(f"✅ Created {len(balanced_datasets)} balanced datasets")
    return balanced_datasets

def evaluate_models_single_iteration(df, iteration_num):
    """Train and evaluate models for a single iteration"""
    
    stances = [("All_Stances", None), ("FAVOR", 1), ("AGAINST", 0), ("NONE", 2)]
    models = {
        'Logistic_Regression': LogisticRegression(max_iter=1000, random_state=42),
        'XGBoost': XGBClassifier(eval_metric='logloss', random_state=42, verbosity=0)
    }
    
    feature_cols = [
        c for c in df.columns
        if c not in ['target', 'text', 'stance', 'label', 'dataset', 'topic', 'split', 'index']
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    
    results = []
    
    for stance_name, stance_val in stances:
        df_stance = df if stance_val is None else df[df['stance'] == stance_val]
        
        if df_stance['label'].nunique() < 2:
            print(f"  ⚠️ Skipping {stance_name} - insufficient class diversity")
            continue
            
        print(f"  Processing {stance_name} (n={len(df_stance)})...")
        
        X = df_stance[feature_cols]
        y = df_stance['label']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Preprocessing
        imputer = SimpleImputer(strategy='mean')
        scaler = StandardScaler()
        
        X_train_processed = scaler.fit_transform(imputer.fit_transform(X_train))
        X_test_processed = scaler.transform(imputer.transform(X_test))
        
        for model_name, model in models.items():
            try:
                # Train model
                model.fit(X_train_processed, y_train)
                
                # Predict and calculate F1 score
                y_pred = model.predict(X_test_processed)
                f1 = f1_score(y_test, y_pred, average='binary')
                
                # Store result
                results.append({
                    'iteration': iteration_num,
                    'stance': stance_name,
                    'model': model_name,
                    'f1_score': f1,
                    'train_size': len(X_train),
                    'test_size': len(X_test),
                    'class_distribution': f"{y.sum()}/{len(y)-y.sum()}"
                })
                
                print(f"    ✅ {model_name}: F1 = {f1:.4f}")
                
            except Exception as e:
                print(f"    ❌ Error with {model_name}: {e}")
                continue
    
    return results

def run_performance_analysis(base_dir, total_iterations=13):
    """Run complete performance analysis across all iterations"""
    
    print("="*80)
    print("MODEL PERFORMANCE ANALYSIS: F1 SCORE EVALUATION")
    print("="*80)
    
    # Step 1: Load data
    print(f"\n{'='*60}")
    print("STEP 1: LOADING DATA")
    print(f"{'='*60}")
    df_correct, df_mis = load_and_prepare_data(base_dir)
    
    # Step 2: Create balanced datasets
    print(f"\n{'='*60}")
    print("STEP 2: CREATING BALANCED DATASETS")
    print(f"{'='*60}")
    balanced_datasets = create_balanced_datasets_all_data(df_correct, df_mis)
    
    # Step 3: Run evaluation for each iteration
    print(f"\n{'='*60}")
    print("STEP 3: EVALUATING MODELS ACROSS ITERATIONS")
    print(f"{'='*60}")
    
    all_results = []
    
    for i, df in enumerate(balanced_datasets, 1):
        print(f"\nIteration {i}/{total_iterations}...")
        iteration_results = evaluate_models_single_iteration(df, i)
        all_results.extend(iteration_results)
    
    # Step 4: Calculate summary statistics
    print(f"\n{'='*60}")
    print("STEP 4: CALCULATING SUMMARY STATISTICS")
    print(f"{'='*60}")
    
    results_df = pd.DataFrame(all_results)
    
    if len(results_df) == 0:
        print("❌ No results to analyze")
        return
    
    # Calculate mean and std F1 scores by stance and model
    summary_stats = results_df.groupby(['stance', 'model'])['f1_score'].agg(['mean', 'std', 'count']).round(4)
    summary_stats.columns = ['mean_f1', 'std_f1', 'n_iterations']
    summary_stats = summary_stats.reset_index()
    
    # Calculate overall average for each stance
    stance_averages = results_df.groupby('stance')['f1_score'].mean().round(4)
    
    # Create formatted output
    print(f"\n{'='*80}")
    print("FINAL RESULTS: F1 SCORE SUMMARY")
    print(f"{'='*80}")
    print(f"{'Stance':<12} {'LR Mean':<10} {'LR Std':<10} {'XGB Mean':<10} {'XGB Std':<10} {'Average':<10}")
    print("-" * 80)
    
    for stance in ['All_Stances', 'FAVOR', 'AGAINST', 'NONE']:
        lr_data = summary_stats[(summary_stats['stance'] == stance) & (summary_stats['model'] == 'Logistic_Regression')]
        xgb_data = summary_stats[(summary_stats['stance'] == stance) & (summary_stats['model'] == 'XGBoost')]
        
        lr_mean = lr_data['mean_f1'].iloc[0] if len(lr_data) > 0 else 0.0
        lr_std = lr_data['std_f1'].iloc[0] if len(lr_data) > 0 else 0.0
        xgb_mean = xgb_data['mean_f1'].iloc[0] if len(xgb_data) > 0 else 0.0
        xgb_std = xgb_data['std_f1'].iloc[0] if len(xgb_data) > 0 else 0.0
        avg = stance_averages.get(stance, 0.0)
        
        print(f"{stance:<12} {lr_mean:<10.4f} {lr_std:<10.4f} {xgb_mean:<10.4f} {xgb_std:<10.4f} {avg:<10.4f}")
    
    # Save detailed results into the results/ directory.
    detailed_path = RESULTS_DIR / 'model_performance_detailed.csv'
    summary_path = RESULTS_DIR / 'model_performance_summary.csv'
    results_df.to_csv(detailed_path, index=False)
    summary_stats.to_csv(summary_path, index=False)

    print(f"\n✅ Analysis complete!")
    print(f"📁 Detailed results saved to: {detailed_path}")
    print(f"📊 Summary statistics saved to: {summary_path}")
    print(f"📈 Total evaluations: {len(results_df)}")

    return results_df, summary_stats

if __name__ == "__main__":
    base_dir = RESULTS_DIR / "evaluation_results"
    results_df, summary_stats = run_performance_analysis(str(base_dir))
