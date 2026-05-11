import pandas as pd
import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

# ============================================================================
# PART 1: DATA LOADING AND DATASET CREATION
# ============================================================================

def load_and_prepare_data(base_dir):
    """Load data and return separate correct and misclassified dataframes for multiple undersampling"""
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

# ============================================================================
# PART 2: SHAP ANALYSIS FUNCTIONS
# ============================================================================

def calculate_shap_values(model, X_train, X_test, model_type='linear'):
    """Calculate SHAP values for a given model"""
    
    if model_type == 'linear':
        # Use LinearExplainer for logistic regression
        explainer = shap.LinearExplainer(model, X_train)
        shap_values = explainer.shap_values(X_test)
    elif model_type == 'tree':
        # Use TreeExplainer for XGBoost
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
    else:
        # Use KernelExplainer as fallback (slower but works for any model)
        explainer = shap.KernelExplainer(model.predict_proba, X_train.sample(100))
        shap_values = explainer.shap_values(X_test)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Take positive class for binary classification
    
    return explainer, shap_values

def aggregate_shap_importance(shap_values, feature_names):
    """Calculate global feature importance from SHAP values"""
    
    # Mean absolute SHAP values (global importance)
    shap_importance = np.abs(shap_values).mean(axis=0)
    
    # Create DataFrame
    shap_df = pd.DataFrame({
        'feature': feature_names,
        'shap_importance': shap_importance
    }).sort_values('shap_importance', ascending=False)
    
    return shap_df

def run_shap_analysis_single_iteration(df, output_dir, iteration_num):
    """Run SHAP analysis for a single iteration"""
    
    stances = [("All_Stances", None), ("FAVOR", 1), ("AGAINST", 0), ("NONE", 2)]
    models = {
        'Logistic Regression': (LogisticRegression(max_iter=1000), 'linear'),
        'XGBoost': (XGBClassifier(eval_metric='logloss'), 'tree')
    }
    
    # Create iteration-specific output directory
    iteration_dir = os.path.join(output_dir, f"{iteration_num:02d}_iteration_shap")
    os.makedirs(iteration_dir, exist_ok=True)
    
    feature_cols = [
        c for c in df.columns
        if c not in ['target', 'text', 'stance', 'label', 'dataset', 'topic', 'split', 'index']
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    
    all_results = []
    
    for stance_name, stance_val in stances:
        df_stance = df if stance_val is None else df[df['stance'] == stance_val]
        
        if df_stance['label'].nunique() < 2:
            continue
            
        print(f"  Processing {stance_name} (n={len(df_stance)})...")
        
        X = df_stance[feature_cols]
        y = df_stance['label']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        # Preprocessing
        imputer = SimpleImputer(strategy='mean')
        scaler = StandardScaler()
        
        X_train_processed = scaler.fit_transform(imputer.fit_transform(X_train))
        X_test_processed = scaler.transform(imputer.transform(X_test))
        
        X_train_df = pd.DataFrame(X_train_processed, columns=feature_cols)
        X_test_df = pd.DataFrame(X_test_processed, columns=feature_cols)
        
        for model_name, (model, model_type) in models.items():
            try:
                # Train model
                model.fit(X_train_processed, y_train)
                
                # Calculate SHAP values
                explainer, shap_values = calculate_shap_values(model, X_train_df, X_test_df, model_type)
                
                # Get global importance
                shap_importance_df = aggregate_shap_importance(shap_values, feature_cols)
                shap_importance_df['iteration'] = iteration_num
                shap_importance_df['stance'] = stance_name
                shap_importance_df['model'] = model_name
                
                # Save results
                output_file = os.path.join(iteration_dir, f"shap_importance_{stance_name}_{model_name.replace(' ', '_')}.csv")
                shap_importance_df.to_csv(output_file, index=False)
                
                # Add to all results
                all_results.append(shap_importance_df)
                
                # Create SHAP plots (for first few features to avoid clutter)
                try:
                    plt.figure(figsize=(10, 6))
                    shap.summary_plot(shap_values[:100], X_test_df.iloc[:100], 
                                    feature_names=feature_cols, show=False, max_display=10)
                    plt.title(f'SHAP Summary - {stance_name} - {model_name} (Iteration {iteration_num})')
                    plt.tight_layout()
                    plot_file = os.path.join(iteration_dir, f"shap_summary_{stance_name}_{model_name.replace(' ', '_')}.png")
                    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    # Feature importance plot
                    plt.figure(figsize=(10, 8))
                    top_features = shap_importance_df.head(15)
                    plt.barh(range(len(top_features)), top_features['shap_importance'])
                    plt.yticks(range(len(top_features)), top_features['feature'])
                    plt.xlabel('Mean |SHAP Value|')
                    plt.title(f'Top 15 Features by SHAP Importance\n{stance_name} - {model_name} (Iteration {iteration_num})')
                    plt.gca().invert_yaxis()
                    plt.tight_layout()
                    importance_plot_file = os.path.join(iteration_dir, f"shap_feature_importance_{stance_name}_{model_name.replace(' ', '_')}.png")
                    plt.savefig(importance_plot_file, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                except Exception as e:
                    print(f"    Warning: Could not create SHAP plots for {stance_name}-{model_name}: {e}")
                
                print(f"    ✅ {model_name} completed for {stance_name}")
                
            except Exception as e:
                print(f"    ❌ Error with {model_name} for {stance_name}: {e}")
                continue
    
    return all_results

def aggregate_shap_across_iterations(base_dir, output_dir, total_iterations=13, min_occurrences=7):
    """Aggregate SHAP importance across multiple iterations"""
    
    print(f"\n{'='*60}")
    print("AGGREGATING SHAP RESULTS ACROSS ITERATIONS")
    print(f"{'='*60}")
    
    stances = ["All_Stances", "FAVOR", "AGAINST", "NONE"]
    models = ["Logistic Regression", "XGBoost"]
    
    final_output_dir = os.path.join(output_dir, "final_shap_aggregated")
    os.makedirs(final_output_dir, exist_ok=True)
    
    for stance in stances:
        for model_name in models:
            print(f"\nAggregating {model_name} results for {stance}...")
            
            # Collect SHAP values from all iterations
            all_shap_data = []
            
            for i in range(1, total_iterations + 1):
                file_path = os.path.join(base_dir, f"{i:02d}_iteration_shap", 
                                       f"shap_importance_{stance}_{model_name.replace(' ', '_')}.csv")
                
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    df['iteration'] = i
                    all_shap_data.append(df)
            
            if not all_shap_data:
                print(f"  No data found for {stance}-{model_name}")
                continue
            
            # Combine all iterations
            combined_df = pd.concat(all_shap_data, ignore_index=True)
            
            # Aggregate by feature
            feature_aggregated = combined_df.groupby('feature').agg({
                'shap_importance': ['mean', 'std', 'count'],
                'iteration': lambda x: list(x)
            }).round(6)
            
            feature_aggregated.columns = ['mean_shap_importance', 'std_shap_importance', 'num_iterations', 'iterations_list']
            feature_aggregated = feature_aggregated.reset_index()
            
            # Filter by minimum occurrences
            feature_aggregated = feature_aggregated[feature_aggregated['num_iterations'] >= min_occurrences]
            
            # Sort by mean importance
            feature_aggregated = feature_aggregated.sort_values('mean_shap_importance', ascending=False)
            
            # Add additional metrics
            feature_aggregated['coefficient_of_variation'] = (feature_aggregated['std_shap_importance'] / 
                                                            feature_aggregated['mean_shap_importance']).round(4)
            feature_aggregated['consistency_score'] = (feature_aggregated['num_iterations'] / total_iterations * 
                                                     feature_aggregated['mean_shap_importance']).round(6)
            
            # Save results
            output_file = os.path.join(final_output_dir, f"final_shap_{stance}_{model_name.replace(' ', '_')}.csv")
            feature_aggregated.to_csv(output_file, index=False)
            
            print(f"  ✅ Saved {len(feature_aggregated)} features to {output_file}")
            
            # Create visualization
            if len(feature_aggregated) > 0:
                plt.figure(figsize=(12, 8))
                top_features = feature_aggregated.head(20)
                
                plt.errorbar(range(len(top_features)), top_features['mean_shap_importance'], 
                           yerr=top_features['std_shap_importance'], fmt='o', capsize=5, capthick=2)
                
                plt.xticks(range(len(top_features)), top_features['feature'], rotation=45, ha='right')
                plt.ylabel('Mean SHAP Importance')
                plt.title(f'Top 20 Features by SHAP Importance\n{stance} - {model_name} (Aggregated across {total_iterations} iterations)')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                plot_file = os.path.join(final_output_dir, f"final_shap_plot_{stance}_{model_name.replace(' ', '_')}.png")
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plt.close()

# ============================================================================
# PART 3: COMPARISON WITH LINEAR MODEL RESULTS
# ============================================================================

def compare_shap_with_existing_linear_results(shap_dir, linear_base_dir, output_dir):
    """Compare SHAP with existing linear model results from iteration folders"""
    
    print(f"\n{'='*60}")
    print("COMPARING SHAP WITH EXISTING LINEAR MODEL RESULTS")
    print(f"{'='*60}")
    
    stances = ["All_Stances", "FAVOR", "AGAINST", "NONE"]
    models = [("Logistic_Regression", "Logistic Regression"), ("XGBoost", "XGBoost")]
    
    comparison_output_dir = os.path.join(output_dir, "shap_linear_comparison_final")
    os.makedirs(comparison_output_dir, exist_ok=True)
    
    all_correlations = []
    
    for stance in stances:
        for model_file_name, model_display_name in models:
            print(f"\nComparing {stance} - {model_display_name}...")
            
            # Load SHAP results
            shap_file = os.path.join(shap_dir, f"final_shap_{stance}_{model_file_name}.csv")
            if not os.path.exists(shap_file):
                print(f"  SHAP file not found: {shap_file}")
                continue
            
            shap_df = pd.read_csv(shap_file)[['feature', 'mean_shap_importance']].rename(
                columns={'mean_shap_importance': 'shap_importance'})
            
            # Collect linear model results from iterations
            linear_data = {}
            iterations_found = 0
            
            for i in range(1, 14):  # Check all 13 iterations
                linear_file = os.path.join(linear_base_dir, f"{i:02d}_iteration", f"top_features_{stance}.csv")
                if os.path.exists(linear_file):
                    try:
                        linear_df = pd.read_csv(linear_file)
                        iterations_found += 1
                        
                        for _, row in linear_df.iterrows():
                            feature = str(row['feature']).replace('*', '')  # Remove significance markers
                            
                            # Get coefficient/importance based on model
                            if model_display_name == "Logistic Regression" and model_display_name in linear_df.columns:
                                coef_value = row[model_display_name]
                            elif model_display_name == "XGBoost" and model_display_name in linear_df.columns:
                                coef_value = row[model_display_name]
                            else:
                                continue
                            
                            # Store non-zero values
                            if pd.notna(coef_value) and coef_value != 0:
                                if feature not in linear_data:
                                    linear_data[feature] = []
                                linear_data[feature].append(abs(float(coef_value)))
                                
                    except Exception as e:
                        print(f"    Warning: Error reading {linear_file}: {e}")
            
            print(f"  Found linear data from {iterations_found} iterations")
            
            if not linear_data:
                print(f"  No linear model data found for {stance} - {model_display_name}")
                continue
            
            # Calculate mean linear importance for each feature
            linear_summary = []
            for feature, values in linear_data.items():
                if len(values) >= 7:  # Same threshold as your other analyses
                    linear_summary.append({
                        'feature': feature,
                        'linear_importance': np.mean(values),
                        'linear_std': np.std(values),
                        'linear_count': len(values)
                    })
            
            if not linear_summary:
                print(f"  No features meet minimum occurrence threshold for {stance} - {model_display_name}")
                continue
            
            linear_df = pd.DataFrame(linear_summary)
            
            # Merge SHAP and linear results
            comparison_df = shap_df.merge(linear_df, on='feature', how='outer', indicator=True)
            
            # Add rankings
            comparison_df['shap_rank'] = comparison_df['shap_importance'].rank(ascending=False, na_option='bottom')
            comparison_df['linear_rank'] = comparison_df['linear_importance'].rank(ascending=False, na_option='bottom')
            
            # Calculate correlations for overlapping features
            overlap_df = comparison_df.dropna(subset=['shap_importance', 'linear_importance'])
            
            correlations = {}
            if len(overlap_df) > 2:
                # Spearman correlation for importance values
                corr_importance = overlap_df['shap_importance'].corr(overlap_df['linear_importance'], method='spearman')
                correlations['importance_spearman'] = corr_importance
                
                # Spearman correlation for ranks
                corr_ranks = overlap_df['shap_rank'].corr(overlap_df['linear_rank'], method='spearman')
                correlations['rank_spearman'] = corr_ranks
                
                # Pearson correlation for importance values
                corr_pearson = overlap_df['shap_importance'].corr(overlap_df['linear_importance'], method='pearson')
                correlations['importance_pearson'] = corr_pearson
            
            # Sort by SHAP importance
            comparison_df = comparison_df.sort_values('shap_importance', ascending=False, na_position='last')
            
            # Add interpretation columns
            comparison_df['overlap_status'] = comparison_df['_merge'].map({
                'both': 'Both methods',
                'left_only': 'SHAP only',
                'right_only': 'Linear only'
            })
            
            # Save detailed comparison
            output_file = os.path.join(comparison_output_dir, f"detailed_comparison_{stance}_{model_file_name}.csv")
            comparison_df.to_csv(output_file, index=False)
            
            # Save correlation summary
            corr_file = os.path.join(comparison_output_dir, f"correlations_{stance}_{model_file_name}.txt")
            with open(corr_file, 'w') as f:
                f.write(f"SHAP vs Linear Model Correlations\n")
                f.write(f"Stance: {stance}, Model: {model_display_name}\n")
                f.write("="*50 + "\n")
                f.write(f"Overlapping features: {len(overlap_df)}\n")
                f.write(f"SHAP-only features: {len(comparison_df[comparison_df['_merge'] == 'left_only'])}\n")
                f.write(f"Linear-only features: {len(comparison_df[comparison_df['_merge'] == 'right_only'])}\n\n")
                
                if correlations:
                    f.write("Correlations:\n")
                    for corr_name, corr_value in correlations.items():
                        f.write(f"  {corr_name}: {corr_value:.4f}\n")
                else:
                    f.write("No correlations calculated (insufficient overlapping features)\n")
            
            # Store for summary
            all_correlations.append({
                'stance': stance,
                'model': model_display_name,
                'overlapping_features': len(overlap_df),
                'shap_only': len(comparison_df[comparison_df['_merge'] == 'left_only']),
                'linear_only': len(comparison_df[comparison_df['_merge'] == 'right_only']),
                **correlations
            })
            
            print(f"  ✅ Saved to {output_file}")
            print(f"  Overlapping features: {len(overlap_df)}")
            if correlations:
                print(f"  Importance correlation: {correlations.get('importance_spearman', 'N/A'):.4f}")
                print(f"  Rank correlation: {correlations.get('rank_spearman', 'N/A'):.4f}")
            
            # Create visualization for overlapping features
            if len(overlap_df) > 3:
                plt.figure(figsize=(10, 8))
                
                # Scatter plot of importance values
                plt.subplot(2, 1, 1)
                plt.scatter(overlap_df['linear_importance'], overlap_df['shap_importance'], alpha=0.7)
                plt.xlabel(f'{model_display_name} Importance')
                plt.ylabel('SHAP Importance')
                plt.title(f'Importance Correlation: {stance} - {model_display_name}\n' + 
                         f'Spearman r = {correlations.get("importance_spearman", "N/A"):.3f}')
                
                # Add feature labels for top features
                top_features = overlap_df.nlargest(5, 'shap_importance')
                for _, row in top_features.iterrows():
                    plt.annotate(row['feature'], 
                               (row['linear_importance'], row['shap_importance']),
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
                
                # Rank comparison
                plt.subplot(2, 1, 2)
                plt.scatter(overlap_df['linear_rank'], overlap_df['shap_rank'], alpha=0.7)
                plt.xlabel(f'{model_display_name} Rank')
                plt.ylabel('SHAP Rank')
                plt.title(f'Rank Correlation: {stance} - {model_display_name}\n' + 
                         f'Spearman r = {correlations.get("rank_spearman", "N/A"):.3f}')
                
                # Add diagonal line for perfect correlation
                max_rank = max(overlap_df['linear_rank'].max(), overlap_df['shap_rank'].max())
                plt.plot([1, max_rank], [1, max_rank], 'r--', alpha=0.5, label='Perfect correlation')
                plt.legend()
                
                plt.tight_layout()
                plot_file = os.path.join(comparison_output_dir, f"correlation_plot_{stance}_{model_file_name}.png")
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plt.close()
    
    # Save overall summary
    if all_correlations:
        summary_df = pd.DataFrame(all_correlations)
        summary_file = os.path.join(comparison_output_dir, "correlation_summary.csv")
        summary_df.to_csv(summary_file, index=False)
        
        print(f"\n✅ Summary saved to {summary_file}")
        print(f"📊 Generated detailed comparisons for all stance-model combinations")
        
        # Create summary visualization
        create_summary_visualization(comparison_output_dir)
    
    return comparison_output_dir

def create_summary_visualization(comparison_dir):
    """Create summary visualizations of SHAP vs Linear comparisons"""
    
    summary_file = os.path.join(comparison_dir, "correlation_summary.csv")
    if not os.path.exists(summary_file):
        print("No summary file found for visualization")
        return
    
    df = pd.read_csv(summary_file)
    
    # Create summary plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Number of overlapping features
    ax1 = axes[0, 0]
    pivot1 = df.pivot(index='stance', columns='model', values='overlapping_features')
    pivot1.plot(kind='bar', ax=ax1, alpha=0.8)
    ax1.set_title('Number of Overlapping Features')
    ax1.set_ylabel('Count')
    ax1.legend(title='Model')
    plt.setp(ax1.get_xticklabels(), rotation=45)
    
    # Plot 2: Importance correlations
    ax2 = axes[0, 1]
    if 'importance_spearman' in df.columns:
        pivot2 = df.pivot(index='stance', columns='model', values='importance_spearman')
        pivot2.plot(kind='bar', ax=ax2, alpha=0.8)
        ax2.set_title('Importance Value Correlations (Spearman)')
        ax2.set_ylabel('Correlation')
        ax2.legend(title='Model')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.setp(ax2.get_xticklabels(), rotation=45)
    
    # Plot 3: Rank correlations
    ax3 = axes[1, 0]
    if 'rank_spearman' in df.columns:
        pivot3 = df.pivot(index='stance', columns='model', values='rank_spearman')
        pivot3.plot(kind='bar', ax=ax3, alpha=0.8)
        ax3.set_title('Rank Correlations (Spearman)')
        ax3.set_ylabel('Correlation')
        ax3.legend(title='Model')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.setp(ax3.get_xticklabels(), rotation=45)
    
    # Plot 4: Feature distribution
    ax4 = axes[1, 1]
    
    # Stack the different types of features
    labels = [f"{stance}\n{model}" for stance, model in zip(df['stance'], df['model'])]
    x_pos = np.arange(len(labels))
    
    ax4.bar(x_pos, df['overlapping_features'], label='Both methods', alpha=0.8)
    ax4.bar(x_pos, df['shap_only'], bottom=df['overlapping_features'], 
           label='SHAP only', alpha=0.8)
    ax4.bar(x_pos, df['linear_only'], 
           bottom=df['overlapping_features'] + df['shap_only'], 
           label='Linear only', alpha=0.8)
    
    ax4.set_title('Feature Distribution by Method')
    ax4.set_ylabel('Number of Features')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(labels, rotation=45, ha='right')
    ax4.legend()
    
    plt.tight_layout()
    plot_file = os.path.join(comparison_dir, "summary_visualization.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Summary visualization saved to {plot_file}")

# ============================================================================
# PART 4: MAIN EXECUTION FUNCTIONS
# ============================================================================

def run_complete_shap_analysis(base_dir, output_dir, linear_iterations_dir, total_iterations=13):
    """Run complete SHAP analysis pipeline with comparison to existing linear results"""
    
    print("🚀 STARTING COMPREHENSIVE SHAP ANALYSIS WITH LINEAR COMPARISON")
    print("="*80)
    
    # Step 1: Load data
    print("\n" + "="*60)
    print("STEP 1: LOADING DATA")
    print("="*60)
    df_correct, df_mis = load_and_prepare_data(base_dir)
    
    # Step 2: Create balanced datasets
    print("\n" + "="*60)
    print("STEP 2: CREATING BALANCED DATASETS")
    print("="*60)
    balanced_datasets = create_balanced_datasets_all_data(df_correct, df_mis)
    
    # Step 3: Run SHAP analysis for each iteration
    print(f"\n{'='*60}")
    print("STEP 3: RUNNING SHAP ANALYSIS FOR EACH ITERATION")
    print(f"{'='*60}")
    
    for i, df in enumerate(balanced_datasets, 1):
        print(f"\nIteration {i}/{total_iterations}...")
        run_shap_analysis_single_iteration(df, output_dir, i)
    
    # Step 4: Aggregate results across iterations
    print(f"\n{'='*60}")
    print("STEP 4: AGGREGATING SHAP RESULTS ACROSS ITERATIONS")
    print(f"{'='*60}")
    aggregate_shap_across_iterations(output_dir, output_dir, total_iterations)
    
    # Step 5: Compare with existing linear model results
    print(f"\n{'='*60}")
    print("STEP 5: COMPARING SHAP WITH EXISTING LINEAR MODEL RESULTS")
    print(f"{'='*60}")
    shap_dir = os.path.join(output_dir, "final_shap_aggregated")
    comparison_dir = compare_shap_with_existing_linear_results(shap_dir, linear_iterations_dir, output_dir)
    
    print(f"\n🎉 COMPLETE SHAP ANALYSIS FINISHED!")
    print(f"📁 Results saved in: {output_dir}")
    print(f"📊 Comparisons saved in: {comparison_dir}")
    
    return output_dir, comparison_dir

# Quick comparison function (if you already have SHAP results)
def run_comparison_only(shap_dir, linear_iterations_dir, output_dir):
    """Run only the comparison between existing SHAP and linear results"""
    
    print("🔍 RUNNING SHAP VS LINEAR COMPARISON ONLY")
    print("="*60)
    
    comparison_dir = compare_shap_with_existing_linear_results(shap_dir, linear_iterations_dir, output_dir)
    
    print(f"\n✅ Comparison completed!")
    print(f"📊 Results saved in: {comparison_dir}")
    
    return comparison_dir

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Configuration
    base_dir = "evaluation_results"  # Original data directory
    output_dir = "comprehensive_shap_analysis"  # Where all SHAP results will be saved
    linear_iterations_dir = "iterations"  # Where your existing linear model results are
    
    print("Choose analysis type:")
    print("1. Complete SHAP analysis + comparison (if you haven't run SHAP yet)")
    print("2. Comparison only (if you already have SHAP results)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        # Run complete analysis
        run_complete_shap_analysis(base_dir, output_dir, linear_iterations_dir)
    elif choice == "2":
        # Run comparison only
        shap_dir = input("Enter path to SHAP results directory: ").strip()
        if not shap_dir:
            shap_dir = "final_analysis_with_shap/final_shap_aggregated"
        
        run_comparison_only(shap_dir, linear_iterations_dir, output_dir)
    else:
        print("Invalid choice. Defaulting to comparison only with existing results.")
        shap_dir = "final_analysis_with_shap/final_shap_aggregated"
        run_comparison_only(shap_dir, linear_iterations_dir, "final_analysis_with_shap")