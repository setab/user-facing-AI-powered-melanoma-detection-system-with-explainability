"""
Visualization script for model comparison results.
Generates publication-quality plots and tables for thesis writeup.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9


def load_results(results_path: str) -> Dict[str, Any]:
    """Load comparison results"""
    with open(results_path, 'r') as f:
        return json.load(f)


def create_comparison_table(results: Dict[str, Any], output_path: Path):
    """Create LaTeX-formatted comparison table"""
    
    # Prepare data for table
    data = []
    for arch, res in results.items():
        mel_metrics = res['melanoma_metrics']
        data.append({
            'Architecture': arch.replace('_', '-').upper(),
            'Accuracy': f"{res['accuracy_calibrated']:.4f}",
            'AUC (Macro)': f"{res['auc_macro']:.4f}",
            'Melanoma AUC': f"{mel_metrics['auc_melanoma']:.4f}",
            'Sensitivity @ 95% Spec': f"{mel_metrics['sensitivity_at_spec95']:.4f}",
            'ECE': f"{res['ece']:.4f}",
            'Brier Score': f"{res['brier_score']:.4f}",
            'Inference Time (ms)': f"{res['inference_time_mean']*1000:.2f}",
        })
    
    df = pd.DataFrame(data)
    
    # Save as CSV
    csv_path = output_path / 'comparison_table.csv'
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved comparison table to {csv_path}")
    
    # Save as LaTeX
    latex_path = output_path / 'comparison_table.tex'
    latex_str = df.to_latex(index=False, escape=False, column_format='l' + 'c' * (len(df.columns) - 1))
    with open(latex_path, 'w') as f:
        f.write(latex_str)
    print(f"✅ Saved LaTeX table to {latex_path}")
    
    return df


def plot_training_curves(results: Dict[str, Any], output_path: Path):
    """Plot training and validation curves for all models"""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    for arch, res in results.items():
        history = res['training_history']
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss curves
        axes[0].plot(epochs, history['train_loss'], label=f'{arch} (train)', linestyle='--', alpha=0.7)
        axes[0].plot(epochs, history['val_loss'], label=f'{arch} (val)', linewidth=2)
        
        # Accuracy curves
        axes[1].plot(epochs, history['train_acc'], label=f'{arch} (train)', linestyle='--', alpha=0.7)
        axes[1].plot(epochs, history['val_acc'], label=f'{arch} (val)', linewidth=2)
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = output_path / 'training_curves.png'
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved training curves to {save_path}")


def plot_metrics_comparison(results: Dict[str, Any], output_path: Path):
    """Plot bar charts comparing key metrics"""
    
    architectures = list(results.keys())
    
    # Prepare data
    metrics = {
        'Accuracy': [results[a]['accuracy_calibrated'] for a in architectures],
        'AUC (Macro)': [results[a]['auc_macro'] for a in architectures],
        'Melanoma AUC': [results[a]['melanoma_metrics']['auc_melanoma'] for a in architectures],
        'Sensitivity\n@ 95% Spec': [results[a]['melanoma_metrics']['sensitivity_at_spec95'] for a in architectures],
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    colors = plt.cm.Set3(range(len(architectures)))
    
    for idx, (metric_name, values) in enumerate(metrics.items()):
        ax = axes[idx]
        bars = ax.bar(range(len(architectures)), values, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_xticks(range(len(architectures)))
        ax.set_xticklabels([a.replace('_', '-').upper() for a in architectures], rotation=45, ha='right')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} Comparison')
        ax.set_ylim([0, 1.0])
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    save_path = output_path / 'metrics_comparison.png'
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved metrics comparison to {save_path}")


def plot_calibration_comparison(results: Dict[str, Any], output_path: Path):
    """Plot calibration metrics comparison"""
    
    architectures = list(results.keys())
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # ECE comparison
    ece_values = [results[a]['ece'] for a in architectures]
    colors = plt.cm.Set2(range(len(architectures)))
    bars = axes[0].bar(range(len(architectures)), ece_values, color=colors, edgecolor='black', linewidth=0.5)
    axes[0].set_xticks(range(len(architectures)))
    axes[0].set_xticklabels([a.replace('_', '-').upper() for a in architectures], rotation=45, ha='right')
    axes[0].set_ylabel('Expected Calibration Error (ECE)')
    axes[0].set_title('Calibration Error Comparison (Lower is Better)')
    axes[0].grid(True, axis='y', alpha=0.3)
    
    for bar, val in zip(bars, ece_values):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=8)
    
    # Brier Score comparison
    brier_values = [results[a]['brier_score'] for a in architectures]
    bars = axes[1].bar(range(len(architectures)), brier_values, color=colors, edgecolor='black', linewidth=0.5)
    axes[1].set_xticks(range(len(architectures)))
    axes[1].set_xticklabels([a.replace('_', '-').upper() for a in architectures], rotation=45, ha='right')
    axes[1].set_ylabel('Brier Score')
    axes[1].set_title('Brier Score Comparison (Lower is Better)')
    axes[1].grid(True, axis='y', alpha=0.3)
    
    for bar, val in zip(bars, brier_values):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    save_path = output_path / 'calibration_comparison.png'
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved calibration comparison to {save_path}")


def plot_inference_time_comparison(results: Dict[str, Any], output_path: Path):
    """Plot inference time comparison"""
    
    architectures = list(results.keys())
    mean_times = [results[a]['inference_time_mean'] * 1000 for a in architectures]  # Convert to ms
    std_times = [results[a]['inference_time_std'] * 1000 for a in architectures]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    colors = plt.cm.Pastel1(range(len(architectures)))
    bars = ax.bar(range(len(architectures)), mean_times, yerr=std_times, 
                  color=colors, edgecolor='black', linewidth=0.5, capsize=5)
    
    ax.set_xticks(range(len(architectures)))
    ax.set_xticklabels([a.replace('_', '-').upper() for a in architectures], rotation=45, ha='right')
    ax.set_ylabel('Inference Time (ms)')
    ax.set_title('Inference Time Comparison (Lower is Better)')
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add value labels
    for bar, mean_val, std_val in zip(bars, mean_times, std_times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std_val + 0.5,
               f'{mean_val:.2f}±{std_val:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    save_path = output_path / 'inference_time_comparison.png'
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved inference time comparison to {save_path}")


def plot_confusion_matrices(results: Dict[str, Any], output_path: Path, label_map: Dict[str, int]):
    """Plot confusion matrices for all models"""
    
    architectures = list(results.keys())
    n_models = len(architectures)
    
    # Invert label map for class names
    idx_to_label = {v: k for k, v in label_map.items()}
    class_names = [idx_to_label[i] for i in range(len(idx_to_label))]
    
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
    if n_models == 1:
        axes = [axes]
    
    for idx, (arch, ax) in enumerate(zip(architectures, axes)):
        cm = np.array(results[arch]['confusion_matrix'])
        
        # Normalize confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        im = ax.imshow(cm_norm, cmap='Blues', aspect='auto', vmin=0, vmax=1)
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(class_names, fontsize=8)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'{arch.replace("_", "-").upper()}')
        
        # Add text annotations
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                text = ax.text(j, i, f'{cm_norm[i, j]:.2f}',
                             ha="center", va="center", color="white" if cm_norm[i, j] > 0.5 else "black",
                             fontsize=7)
    
    plt.tight_layout()
    save_path = output_path / 'confusion_matrices.png'
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved confusion matrices to {save_path}")


def create_summary_report(results: Dict[str, Any], output_path: Path):
    """Create a comprehensive text summary report"""
    
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("MODEL COMPARISON SUMMARY REPORT")
    report_lines.append("="*80)
    report_lines.append("")
    
    # Overall comparison
    report_lines.append("## OVERALL PERFORMANCE RANKING")
    report_lines.append("")
    
    # Rank by accuracy
    accuracy_ranking = sorted(results.items(), 
                             key=lambda x: x[1]['accuracy_calibrated'], reverse=True)
    report_lines.append("### By Accuracy:")
    for rank, (arch, res) in enumerate(accuracy_ranking, 1):
        report_lines.append(f"  {rank}. {arch.upper():<20} {res['accuracy_calibrated']:.4f}")
    report_lines.append("")
    
    # Rank by melanoma AUC
    auc_ranking = sorted(results.items(),
                        key=lambda x: x[1]['melanoma_metrics']['auc_melanoma'], reverse=True)
    report_lines.append("### By Melanoma AUC:")
    for rank, (arch, res) in enumerate(auc_ranking, 1):
        report_lines.append(f"  {rank}. {arch.upper():<20} {res['melanoma_metrics']['auc_melanoma']:.4f}")
    report_lines.append("")
    
    # Rank by calibration (ECE - lower is better)
    ece_ranking = sorted(results.items(), key=lambda x: x[1]['ece'])
    report_lines.append("### By Calibration Quality (ECE - lower is better):")
    for rank, (arch, res) in enumerate(ece_ranking, 1):
        report_lines.append(f"  {rank}. {arch.upper():<20} {res['ece']:.4f}")
    report_lines.append("")
    
    # Rank by inference speed (lower is better)
    speed_ranking = sorted(results.items(), key=lambda x: x[1]['inference_time_mean'])
    report_lines.append("### By Inference Speed (lower is better):")
    for rank, (arch, res) in enumerate(speed_ranking, 1):
        time_ms = res['inference_time_mean'] * 1000
        report_lines.append(f"  {rank}. {arch.upper():<20} {time_ms:.2f} ms")
    report_lines.append("")
    
    # Detailed metrics for each model
    report_lines.append("="*80)
    report_lines.append("DETAILED METRICS BY MODEL")
    report_lines.append("="*80)
    report_lines.append("")
    
    for arch, res in results.items():
        report_lines.append(f"## {arch.upper()}")
        report_lines.append("-" * 40)
        report_lines.append(f"  Overall Accuracy:        {res['accuracy_calibrated']:.4f}")
        report_lines.append(f"  AUC (Macro):             {res['auc_macro']:.4f}")
        report_lines.append(f"  Expected Calib Error:    {res['ece']:.4f}")
        report_lines.append(f"  Brier Score:             {res['brier_score']:.4f}")
        report_lines.append(f"  Temperature:             {res['temperature']:.4f}")
        report_lines.append(f"  Inference Time:          {res['inference_time_mean']*1000:.2f} ± {res['inference_time_std']*1000:.2f} ms")
        report_lines.append("")
        report_lines.append("  Melanoma-Specific Metrics:")
        mel = res['melanoma_metrics']
        report_lines.append(f"    AUC:                   {mel['auc_melanoma']:.4f}")
        report_lines.append(f"    Threshold @ 95% Spec:  {mel['threshold_spec95']:.4f}")
        report_lines.append(f"    Sensitivity @ 95% Spec:{mel['sensitivity_at_spec95']:.4f}")
        report_lines.append(f"    PPV @ 95% Spec:        {mel['ppv_at_spec95']:.4f}")
        report_lines.append(f"    NPV @ 95% Spec:        {mel['npv_at_spec95']:.4f}")
        report_lines.append("")
    
    # Recommendations
    report_lines.append("="*80)
    report_lines.append("RECOMMENDATIONS")
    report_lines.append("="*80)
    report_lines.append("")
    
    best_acc = accuracy_ranking[0][0]
    best_auc = auc_ranking[0][0]
    best_cal = ece_ranking[0][0]
    best_speed = speed_ranking[0][0]
    
    report_lines.append(f"✅ Best Overall Accuracy:    {best_acc.upper()}")
    report_lines.append(f"✅ Best Melanoma Detection:  {best_auc.upper()}")
    report_lines.append(f"✅ Best Calibration:         {best_cal.upper()}")
    report_lines.append(f"✅ Fastest Inference:        {best_speed.upper()}")
    report_lines.append("")
    
    if best_acc == best_auc == best_cal:
        report_lines.append(f"🎯 RECOMMENDED MODEL: {best_acc.upper()}")
        report_lines.append("   This model performs best across multiple criteria.")
    else:
        report_lines.append("🎯 RECOMMENDED MODEL DEPENDS ON PRIORITY:")
        report_lines.append(f"   - For highest accuracy and melanoma detection: {best_auc.upper()}")
        report_lines.append(f"   - For best probability calibration: {best_cal.upper()}")
        report_lines.append(f"   - For fastest deployment: {best_speed.upper()}")
    report_lines.append("")
    
    # Save report
    report_path = output_path / 'summary_report.txt'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"✅ Saved summary report to {report_path}")
    
    # Also print to console
    print("\n" + '\n'.join(report_lines))


def main():
    parser = argparse.ArgumentParser(description='Visualize model comparison results')
    parser.add_argument('--results', required=True, help='Path to comparison_results.json')
    parser.add_argument('--label-map', default='models/label_maps/label_map_nb.json')
    parser.add_argument('--output-dir', help='Output directory (defaults to same as results)')
    args = parser.parse_args()
    
    results_path = Path(args.results)
    output_dir = Path(args.output_dir) if args.output_dir else results_path.parent / 'visualizations'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📊 Loading results from {results_path}")
    results = load_results(results_path)
    
    print(f"📁 Output directory: {output_dir}")
    print()
    
    # Load label map
    with open(args.label_map, 'r') as f:
        label_map = json.load(f)
    
    # Generate all visualizations
    print("🎨 Generating visualizations...")
    print()
    
    create_comparison_table(results, output_dir)
    plot_training_curves(results, output_dir)
    plot_metrics_comparison(results, output_dir)
    plot_calibration_comparison(results, output_dir)
    plot_inference_time_comparison(results, output_dir)
    plot_confusion_matrices(results, output_dir, label_map)
    create_summary_report(results, output_dir)
    
    print()
    print("="*80)
    print(f"✅ All visualizations saved to {output_dir}")
    print("="*80)
    print()
    print("📋 Generated files:")
    for file in sorted(output_dir.glob('*')):
        print(f"   - {file.name}")


if __name__ == '__main__':
    main()
