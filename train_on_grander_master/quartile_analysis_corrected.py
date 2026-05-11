#!/usr/bin/env python3
"""
Cumulative Quartile Analysis: Hierarchical LR and SHAP Rank Assignment
======================================================================

This script uses the existing comprehensive analysis results to create:
1. Separate correlation plots (Importance vs Rank correlation) - fixed axis orientation
2. Cumulative quartile analysis with hierarchical exclusion
3. Unified CSV with all features and SHAP directional metrics
4. Focus only on Logistic Regression (exclude XGBoost)

Quartile Assignment (cumulative with exclusions):
- Q1: Both LR and SHAP ranks 1-11 (highest priority)
- Q2: Both ranks 1-22, excluding features already in Q1
- Q3: Both ranks 1-33, excluding features already in Q1 and Q2  
- Q4: Both ranks 1-44, excluding features already in Q1, Q2, and Q3
- Mixed: Features that don't meet any quartile criteria
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import spearmanr, pearsonr

# Import feature name mapping
from feature_label_map import raw_feature_names

# Set style for publication-quality plots
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16
})

def load_existing_comparison_data(comparison_dir, shap_dir, stance):
    """Load existing comparison data and merge with SHAP directional metrics and feature-value relationships"""
    
    print(f"\nLoading existing data for {stance}...")
    
    # Load existing comparison data (has all 44 features with ranks)
    comparison_file = os.path.join(comparison_dir, f"detailed_comparison_{stance}_Logistic_Regression.csv")
    
    if not os.path.exists(comparison_file):
        print(f"  ❌ Comparison file not found: {comparison_file}")
        return None
        
    comparison_df = pd.read_csv(comparison_file)
    print(f"  ✅ Loaded comparison data: {len(comparison_df)} features")
    
    # Load SHAP directional data
    shap_file = os.path.join(shap_dir, f"final_shap_enhanced_{stance}_Logistic_Regression.csv")
    
    if not os.path.exists(shap_file):
        print(f"  ❌ SHAP file not found: {shap_file}")
        return None
        
    shap_df = pd.read_csv(shap_file)[['feature', 'mean_directional', 'positive_ratio', 'negative_ratio', 'direction_strength']]
    print(f"  ✅ Loaded SHAP directional data: {len(shap_df)} features")
    
    # Load feature-value relationship data (NEW)
    feature_value_dir = os.path.join(os.path.dirname(shap_dir), "feature_value_analysis")
    feature_value_file = os.path.join(feature_value_dir, f"feature_value_relationships_{stance}_Logistic_Regression.csv")
    
    if os.path.exists(feature_value_file):
        feature_value_df = pd.read_csv(feature_value_file)[['feature', 'mean_feat_when_shap_positive', 'mean_feat_when_shap_negative', 
                                                           'feature_shap_correlation', 'count_shap_positive', 'count_shap_negative']]
        print(f"  ✅ Loaded feature-value relationships: {len(feature_value_df)} features")
    else:
        print(f"  ⚠️  Feature-value relationships not found: {feature_value_file}")
        print(f"      Run SHAP analysis first to generate this data")
        # Create empty DataFrame with expected columns
        feature_value_df = pd.DataFrame(columns=['feature', 'mean_feat_when_shap_positive', 'mean_feat_when_shap_negative', 
                                               'feature_shap_correlation', 'count_shap_positive', 'count_shap_negative'])
    
    # Merge comparison data with SHAP directional metrics
    merged_df = pd.merge(comparison_df, shap_df, on='feature', how='left')
    print(f"  ✅ Merged comparison and SHAP data: {len(merged_df)} features")
    
    # Merge with feature-value relationships
    merged_df = pd.merge(merged_df, feature_value_df, on='feature', how='left')
    print(f"  ✅ Merged with feature-value data: {len(merged_df)} features")
    
    # Filter to only features present in both methods (overlap_status == 'Both methods')
    both_methods_df = merged_df[merged_df['overlap_status'] == 'Both methods'].copy()
    print(f"  ✅ Features in both methods: {len(both_methods_df)}")
    
    return both_methods_df

def assign_quartiles(df):
    """Assign features to cumulative quartiles with hierarchical exclusion"""
    
    df = df.copy()
    
    # Initialize quartile assignment
    df['quartile'] = 'Unassigned'
    
    # Hierarchical quartile assignment (cumulative with exclusions)
    # Q1: Both ranks 1-11
    q1_mask = (df['shap_rank'] <= 11) & (df['linear_rank'] <= 11)
    df.loc[q1_mask, 'quartile'] = 'Q1'
    
    # Q2: Both ranks 1-22, but exclude Q1 features
    q2_mask = (df['shap_rank'] <= 22) & (df['linear_rank'] <= 22) & (df['quartile'] == 'Unassigned')
    df.loc[q2_mask, 'quartile'] = 'Q2'
    
    # Q3: Both ranks 1-33, but exclude Q1 and Q2 features  
    q3_mask = (df['shap_rank'] <= 33) & (df['linear_rank'] <= 33) & (df['quartile'] == 'Unassigned')
    df.loc[q3_mask, 'quartile'] = 'Q3'
    
    # Q4: Both ranks 1-44, but exclude Q1, Q2, and Q3 features
    q4_mask = (df['shap_rank'] <= 44) & (df['linear_rank'] <= 44) & (df['quartile'] == 'Unassigned')
    df.loc[q4_mask, 'quartile'] = 'Q4'
    
    # Remaining features are Mixed (don't meet any quartile criteria)
    mixed_mask = df['quartile'] == 'Unassigned'
    df.loc[mixed_mask, 'quartile'] = 'Mixed'
    
    return df

def create_separate_correlation_plots(df, stance, output_dir):
    """Create separate PNG files for Importance and Rank correlation plots"""
    
    print(f"  Creating correlation plots for {stance} ({len(df)} features)...")
    
    # Calculate correlations
    importance_corr_pearson, importance_p_pearson = pearsonr(df['shap_importance'], df['linear_importance'])
    importance_corr_spearman, importance_p_spearman = spearmanr(df['shap_importance'], df['linear_importance'])
    
    # For rank correlation: only Spearman on ranks
    rank_corr_spearman, rank_p_spearman = spearmanr(df['shap_rank'], df['linear_rank'])
    
    # Plot 1: Importance Correlation (separate PNG file)
    plt.figure(figsize=(12, 10))
    scatter1 = plt.scatter(df['linear_importance'], df['shap_importance'], 
                          c=df['shap_rank'], cmap='viridis_r', s=120, alpha=0.9, edgecolors='black', linewidth=0.8)
    plt.xlabel('Linear Model Importance (Mean |Coefficient|)', fontsize=18)
    plt.ylabel('SHAP Importance (Mean |SHAP Value|)', fontsize=18)
    plt.title(f'{stance}: Importance Correlation ({len(df)} features)\\n'
              f'Pearson r = {importance_corr_pearson:.3f} (p = {importance_p_pearson:.3f})\\n'
              f'Spearman ρ = {importance_corr_spearman:.3f} (p = {importance_p_spearman:.3f})', fontsize=20)
    
    # Increase tick label sizes
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    # Add colorbar for ranks
    cbar1 = plt.colorbar(scatter1)
    cbar1.set_label('SHAP Rank', fontsize=14)
    cbar1.ax.tick_params(labelsize=12)
    
    # Add trend line
    z1 = np.polyfit(df['linear_importance'], df['shap_importance'], 1)
    p1 = np.poly1d(z1)
    plt.plot(df['linear_importance'], p1(df['linear_importance']), "r--", alpha=0.8, linewidth=2, label='Trend Line')
    
    # Add feature labels for top/bottom features with larger font
    top_features = df.nsmallest(5, 'shap_rank')  # Top 5 by SHAP rank
    for _, row in top_features.iterrows():
        readable_name = raw_feature_names.get(row['feature'], row['feature'])
        plt.annotate(readable_name, 
                   (row['linear_importance'], row['shap_importance']),
                   xytext=(5, 5), textcoords='offset points', fontsize=12, alpha=0.8,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
    
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # Save importance correlation plot in both formats
    importance_png = os.path.join(output_dir, f'importance_correlation_{stance}_LR_vs_SHAP.png')
    importance_pdf = os.path.join(output_dir, f'importance_correlation_{stance}_LR_vs_SHAP.pdf')
    plt.savefig(importance_png, dpi=600, bbox_inches='tight')
    plt.savefig(importance_pdf, dpi=600, bbox_inches='tight')
    print(f"    ✅ Saved importance correlation: {importance_png} & {importance_pdf}")
    plt.close()
    
    # Plot 2: Rank Correlation (separate PNG file)
    plt.figure(figsize=(14, 10))
    
    # Define quartile colors and labels
    quartile_colors = {
        'Q1': '#1f77b4',  # Blue
        'Q2': '#ff7f0e',  # Orange  
        'Q3': '#2ca02c',  # Green
        'Q4': '#d62728',  # Red
        'Mixed': '#9467bd'  # Purple
    }
    
    # Create scatter plot with quartile-based colors
    for quartile, color in quartile_colors.items():
        quartile_data = df[df['quartile'] == quartile]
        if len(quartile_data) > 0:
            plt.scatter(quartile_data['linear_rank'], quartile_data['shap_rank'],
                       c=color, s=120, alpha=0.9, edgecolors='black', linewidth=0.8,
                       label=f'{quartile} ({len(quartile_data)} features)')
    
    plt.xlabel('Logistic Regression Rank', fontsize=18)
    plt.ylabel('SHAP Rank', fontsize=18)
    plt.title(f'{stance}: Rank Correlation (Spearman ρ = {rank_corr_spearman:.3f})', fontsize=20)
    
    # Increase tick label sizes
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    
    # Keep normal axes orientation - (0,0) at bottom left, rank 1 at bottom left
    # No axis inversion needed
    
    # Add diagonal line for perfect rank correlation
    max_rank = max(df['linear_rank'].max(), df['shap_rank'].max())
    plt.plot([1, max_rank], [1, max_rank], "r--", alpha=0.8, linewidth=2, label='Perfect Agreement')
    
    # Add readable feature names to all dots with larger font
    for _, row in df.iterrows():
        # Get readable name from mapping, fallback to raw name if not found
        readable_name = raw_feature_names.get(row['feature'], row['feature'])
        plt.annotate(readable_name, 
                   (row['linear_rank'], row['shap_rank']),
                   xytext=(8, 8), textcoords='offset points', fontsize=10, alpha=0.9,
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.6, edgecolor='gray', linewidth=0.5))
    
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # Save rank correlation plot in both formats
    rank_png = os.path.join(output_dir, f'rank_correlation_{stance}_LR_vs_SHAP.png')
    rank_pdf = os.path.join(output_dir, f'rank_correlation_{stance}_LR_vs_SHAP.pdf')
    plt.savefig(rank_png, dpi=600, bbox_inches='tight')
    plt.savefig(rank_pdf, dpi=600, bbox_inches='tight')
    print(f"    ✅ Saved rank correlation: {rank_png} & {rank_pdf}")
    plt.close()
    
    return {
        'importance_pearson': importance_corr_pearson,
        'importance_spearman': importance_corr_spearman,
        'rank_spearman': rank_corr_spearman
    }

def create_directional_analysis_plot(df, stance, output_dir):
    """Create comprehensive directional analysis for ALL features"""
    
    # Sort by SHAP importance for consistent ordering (ascending for horizontal bars)
    df_sorted = df.sort_values('shap_importance', ascending=True)
    
    # Create figure with multiple subplots - make it taller to accommodate all features
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(24, max(20, len(df_sorted) * 0.6)))
    
    # Plot 1: Mean Directional Values (ALL features)
    colors = ['red' if x < 0 else 'blue' for x in df_sorted['mean_directional']]
    bars1 = ax1.barh(range(len(df_sorted)), df_sorted['mean_directional'], color=colors, alpha=0.7)
    ax1.set_yticks(range(len(df_sorted)))
    ax1.set_yticklabels(df_sorted['feature'], fontsize=max(8, 300//len(df_sorted)))
    ax1.set_xlabel('Mean Directional SHAP Value')
    ax1.set_title(f'{stance}: Mean Directional Values (All {len(df_sorted)} Features)\\n(Red=Negative, Blue=Positive)')
    ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Positive vs Negative Ratios (ALL features)
    x_pos = np.arange(len(df_sorted))
    ax2.barh(x_pos, df_sorted['positive_ratio'], alpha=0.7, label='Positive Ratio', color='blue')
    ax2.barh(x_pos, -df_sorted['negative_ratio'], alpha=0.7, label='Negative Ratio', color='red')
    ax2.set_yticks(x_pos)
    ax2.set_yticklabels(df_sorted['feature'], fontsize=max(8, 300//len(df_sorted)))
    ax2.set_xlabel('Ratio (Positive → | ← Negative)')
    ax2.set_title(f'{stance}: Positive vs Negative Contribution Ratios (All {len(df_sorted)} Features)')
    ax2.legend()
    ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Direction Strength (ALL features)
    if df_sorted['direction_strength'].max() > 0:
        colors3 = plt.cm.viridis(df_sorted['direction_strength'] / df_sorted['direction_strength'].max())
    else:
        colors3 = ['gray'] * len(df_sorted)
    bars3 = ax3.barh(range(len(df_sorted)), df_sorted['direction_strength'], color=colors3, alpha=0.8)
    ax3.set_yticks(range(len(df_sorted)))
    ax3.set_yticklabels(df_sorted['feature'], fontsize=max(8, 300//len(df_sorted)))
    ax3.set_xlabel('Direction Strength (|mean| / std)')
    ax3.set_title(f'{stance}: Directional Consistency Strength (All {len(df_sorted)} Features)')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Quartile Distribution
    quartile_counts = df['quartile'].value_counts()
    colors4 = ['lightblue', 'lightgreen', 'orange', 'lightcoral', 'gray']
    wedges, texts, autotexts = ax4.pie(quartile_counts.values, labels=quartile_counts.index, 
                                      autopct='%1.1f%%', colors=colors4[:len(quartile_counts)], 
                                      startangle=90)
    ax4.set_title(f'{stance}: Feature Distribution by Quartile Agreement')
    
    # Make pie chart text larger
    for text in texts:
        text.set_fontsize(10)
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_color('white')
        autotext.set_weight('bold')
    
    plt.tight_layout()
    
    # Save plot in both formats
    output_png = os.path.join(output_dir, f'directional_analysis_{stance}_all_{len(df_sorted)}_features.png')
    output_pdf = os.path.join(output_dir, f'directional_analysis_{stance}_all_{len(df_sorted)}_features.pdf')
    plt.savefig(output_png, dpi=600, bbox_inches='tight')
    plt.savefig(output_pdf, dpi=600, bbox_inches='tight')
    print(f"  ✅ Saved directional analysis (all {len(df_sorted)} features): {output_png} & {output_pdf}")
    
    plt.close()

def create_unified_quartile_features_csv(all_results, output_dir):
    """Create single CSV with all unique features and their quartile assignments + SHAP directional metrics"""
    
    print(f"\\n{'='*20} CREATING UNIFIED QUARTILE FEATURES CSV {'='*20}")
    
    all_features_data = []
    
    for stance, df in all_results.items():
        # Add each feature with its metrics
        for _, row in df.iterrows():
            feature_data = {
                'stance': stance,
                'feature': row['feature'],
                'quartile': row['quartile'],  # ONLY quartile column needed
                'shap_rank': row['shap_rank'],
                'linear_rank': row['linear_rank'],
                'shap_importance': row['shap_importance'],
                'linear_importance': row['linear_importance'],
                'mean_directional': row['mean_directional'],
                'positive_ratio': row['positive_ratio'],
                'negative_ratio': row['negative_ratio'],
                'direction_strength': row['direction_strength'],
                # NEW: Feature-value relationship columns
                'mean_feat_when_shap_positive': row.get('mean_feat_when_shap_positive', np.nan),
                'mean_feat_when_shap_negative': row.get('mean_feat_when_shap_negative', np.nan),
                'feature_shap_correlation': row.get('feature_shap_correlation', np.nan),
                'count_shap_positive': row.get('count_shap_positive', np.nan),
                'count_shap_negative': row.get('count_shap_negative', np.nan)
            }
            all_features_data.append(feature_data)
    
    # Create DataFrame
    unified_df = pd.DataFrame(all_features_data)
    
    # Save unified CSV
    unified_file = os.path.join(output_dir, 'unified_quartile_features_all_stances.csv')
    unified_df.to_csv(unified_file, index=False)
    print(f"  ✅ Saved unified features CSV: {unified_file}")
    
    # Create quartile summary statistics
    quartile_summary = []
    
    for stance in unified_df['stance'].unique():
        stance_data = unified_df[unified_df['stance'] == stance]
        
        for quartile in ['Q1', 'Q2', 'Q3', 'Q4', 'Mixed']:
            quartile_data = stance_data[stance_data['quartile'] == quartile]
            
            if len(quartile_data) > 0:
                summary_row = {
                    'stance': stance,
                    'quartile': quartile,
                    'n_features': len(quartile_data),
                    'mean_directional_avg': quartile_data['mean_directional'].mean(),
                    'mean_directional_std': quartile_data['mean_directional'].std(),
                    'positive_ratio_avg': quartile_data['positive_ratio'].mean(),
                    'negative_ratio_avg': quartile_data['negative_ratio'].mean(),
                    'direction_strength_avg': quartile_data['direction_strength'].mean(),
                    'features_list': ', '.join(quartile_data['feature'].tolist())
                }
                quartile_summary.append(summary_row)
    
    # Save quartile summary
    summary_df = pd.DataFrame(quartile_summary)
    summary_file = os.path.join(output_dir, 'quartile_summary_by_stance.csv')
    summary_df.to_csv(summary_file, index=False)
    print(f"  ✅ Saved quartile summary: {summary_file}")
    
    return unified_file, summary_file

def main():
    """Main analysis pipeline using existing comprehensive results"""
    
    # Configuration - use existing comprehensive analysis results
    base_dir = "/home/p/parush/style_markers/train_on_grander_master"
    comparison_dir = os.path.join(base_dir, "comprehensive_shap_analysis/shap_linear_comparison_final")
    shap_dir = os.path.join(base_dir, "comprehensive_shap_analysis/final_shap_aggregated")
    output_dir = os.path.join(base_dir, "quartile_analysis_corrected_final")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*80)
    print("CORRECTED QUARTILE ANALYSIS: LOGISTIC REGRESSION vs SHAP")
    print("Using existing comprehensive analysis results")
    print("="*80)
    
    stances = ["All_Stances", "FAVOR", "AGAINST", "NONE"]
    all_results = {}
    correlation_summary = []
    
    for stance in stances:
        print(f"\\n{'='*20} {stance} {'='*20}")
        
        # Load existing comparison data with SHAP directional metrics
        df = load_existing_comparison_data(comparison_dir, shap_dir, stance)
        
        if df is None:
            print(f"  ❌ Skipping {stance} due to data issues")
            continue
            
        # Assign quartiles
        df = assign_quartiles(df)
        
        # Create separate correlation plots
        correlations = create_separate_correlation_plots(df, stance, output_dir)
        correlation_summary.append({
            'stance': stance,
            **correlations
        })
        
        # Create directional analysis for all features
        create_directional_analysis_plot(df, stance, output_dir)
        
        # Save individual stance CSV with simplified quartile structure including feature-value relationships
        stance_file = os.path.join(output_dir, f'quartile_features_{stance}.csv')
        
        # Define columns to include (handle missing columns gracefully)
        base_columns = ['feature', 'quartile', 'shap_rank', 'linear_rank', 'shap_importance', 'linear_importance', 
                       'mean_directional', 'positive_ratio', 'negative_ratio', 'direction_strength']
        feature_value_columns = ['mean_feat_when_shap_positive', 'mean_feat_when_shap_negative', 
                               'feature_shap_correlation', 'count_shap_positive', 'count_shap_negative']
        
        # Only include columns that exist in the DataFrame
        available_columns = [col for col in base_columns + feature_value_columns if col in df.columns]
        df_simplified = df[available_columns].copy()
        
        df_simplified.to_csv(stance_file, index=False)
        print(f"  ✅ Saved quartile features for {stance}: {stance_file}")
        print(f"      Columns included: {len(available_columns)} ({', '.join(available_columns[:5])}...)")
        
        # Store results
        all_results[stance] = df
    
    # Create cross-stance comparisons
    print(f"\\n{'='*20} CROSS-STANCE ANALYSIS {'='*20}")
    
    # Save correlation summary
    correlation_df = pd.DataFrame(correlation_summary)
    correlation_file = os.path.join(output_dir, 'correlation_summary_all_stances.csv')
    correlation_df.to_csv(correlation_file, index=False)
    print(f"  ✅ Saved correlation summary: {correlation_file}")
    
    # Create unified quartile features CSV
    unified_file, summary_file = create_unified_quartile_features_csv(all_results, output_dir)
    
    # Create summary report
    create_summary_report(all_results, output_dir)
    
    print(f"\\n{'='*80}")
    print("✅ CORRECTED ANALYSIS COMPLETE!")
    print(f"📁 Results saved to: {output_dir}")
    print(f"📊 Total unique features processed: {len(set().union(*[df['feature'].tolist() for df in all_results.values()]))}")
    print(f"{'='*80}")

def create_summary_report(all_results, output_dir):
    """Create a comprehensive summary report"""
    
    report_file = os.path.join(output_dir, 'analysis_summary_report.txt')
    
    with open(report_file, 'w') as f:
        f.write("CORRECTED QUARTILE ANALYSIS: LOGISTIC REGRESSION vs SHAP - SUMMARY REPORT\\n")
        f.write("="*80 + "\\n\\n")
        
        f.write("OVERVIEW:\\n")
        f.write("---------\\n")
        f.write("This analysis uses existing comprehensive SHAP vs LR comparison results\\n")
        f.write("to create quartile-based analysis with proper feature coverage.\\n\\n")
        
        f.write("DATA SOURCE:\\n")
        f.write("------------\\n")
        f.write("• Uses comprehensive_shap_analysis/shap_linear_comparison_final/ results\\n")
        f.write("• Includes all features present in both LR and SHAP methods\\n")
        f.write("• Merges with SHAP directional metrics from final_shap_aggregated/\\n\\n")
        
        f.write("SIMPLIFIED QUARTILE METHODOLOGY:\\n")
        f.write("--------------------------------\\n")
        f.write("• Single 'quartile' column with cumulative logic\\n")
        f.write("• Q1: Both LR and SHAP ranks 1-11 (highest priority)\\n")
        f.write("• Q2: Both ranks 1-22, excluding Q1 features\\n")
        f.write("• Q3: Both ranks 1-33, excluding Q1 and Q2 features\\n")
        f.write("• Q4: Both ranks 1-44, excluding Q1, Q2, and Q3 features\\n")
        f.write("• Mixed: Features that don't meet any quartile criteria\\n\\n")
        
        f.write("FEATURES BY STANCE:\\n")
        f.write("------------------\\n")
        
        for stance, df in all_results.items():
            f.write(f"\\n{stance}: {len(df)} features\\n")
            quartile_dist = df['quartile'].value_counts()
            for quartile, count in quartile_dist.items():
                f.write(f"  {quartile}: {count} features\\n")
        
        f.write("\\n\\nFILES GENERATED:\\n")
        f.write("----------------\\n")
        f.write("PLOTS (per stance):\\n")
        f.write("• importance_correlation_{stance}_LR_vs_SHAP.png - Importance correlation plots\\n")
        f.write("• rank_correlation_{stance}_LR_vs_SHAP.png - Rank correlation plots\\n")
        f.write("• directional_analysis_{stance}_all_features.png - Directional analysis plots\\n\\n")
        f.write("CSV FILES:\\n")
        f.write("• quartile_features_{stance}.csv - Individual stance quartile assignments\\n")
        f.write("• unified_quartile_features_all_stances.csv - All stances combined\\n")
        f.write("• quartile_summary_by_stance.csv - Summary statistics by quartile\\n")
        f.write("• correlation_summary_all_stances.csv - Correlation coefficients\\n\\n")
        
        f.write("INTERPRETATION:\\n")
        f.write("---------------\\n")
        f.write("INDIVIDUAL STANCE FILES: quartile_features_{stance}.csv\\n")
        f.write("• Clean structure with single 'quartile' column: Q1, Q2, Q3, Q4, Mixed\\n")
        f.write("• Contains features and SHAP directional metrics for each stance\\n")
        f.write("• NEW: Feature-value relationships showing mean feature values when SHAP is positive/negative\\n\\n")
        f.write("UNIFIED FILE: unified_quartile_features_all_stances.csv\\n")
        f.write("• All stances combined with 'stance' column\\n")
        f.write("• Q1-Q4 represent cumulative agreement between LR and SHAP ranks\\n")
        f.write("• Mixed shows features where methods disagree on importance\\n")
        f.write("• NEW COLUMNS:\\n")
        f.write("  - mean_feat_when_shap_positive: Mean feature value when SHAP pushes toward correct classification\\n")
        f.write("  - mean_feat_when_shap_negative: Mean feature value when SHAP pushes toward misclassification\\n")
        f.write("  - feature_shap_correlation: Correlation between feature value and SHAP contribution\\n")
        f.write("  - count_shap_positive/negative: Number of observations for each direction\\n")
    
    print(f"  ✅ Saved summary report: {report_file}")

if __name__ == "__main__":
    main()
