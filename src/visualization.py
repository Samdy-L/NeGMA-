"""
Visualization for the community-level fake-information detection experiment.

Every figure produced here is drawn from real experiment output. No simulated,
interpolated, or hard-coded curves are used.

Inputs (all written by `python main.py`):
    data/experiment_results.csv    held-out test-set predictions and ground truth
    data/model_coefficients.json   coefficients of the actually-trained classifier
    data/community_evolution.csv   per-(community, snapshot) feature detail

Run `python main.py` first, then `python src/visualization.py`.
"""
import json
import os

import matplotlib
matplotlib.use('Agg')  # headless backend; this script only writes files
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

# Set style for academic plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 18

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'plots')
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

RESULTS_CSV = os.path.join(DATA_DIR, 'experiment_results.csv')
COEF_JSON = os.path.join(DATA_DIR, 'model_coefficients.json')
EVOLUTION_CSV = os.path.join(DATA_DIR, 'community_evolution.csv')


def _require(path):
    """Fail loudly rather than silently drawing something that isn't measured."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {os.path.relpath(path, BASE_DIR)} - "
            f"run `python main.py` first to generate the experiment data."
        )
    return path


def plot_roc_comparison():
    """
    ROC curve on the held-out test set, computed from the saved predictions.

    Note: the rule stage rarely fires on this dataset, so this curve is in
    practice the logistic-regression branch of the two-stage framework.
    """
    df = pd.read_csv(_require(RESULTS_CSV))
    y_true = df['ground_truth_label'].to_numpy()
    y_score = df['predicted_probability'].to_numpy()

    if len(np.unique(y_true)) < 2:
        print("Test set contains only one class - ROC is undefined, skipping.")
        return

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    n_rule = int(df['rule_flag'].sum())
    print(f"  rule stage fired on {n_rule}/{len(df)} test communities")

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='#d62728', lw=2.5,
             label=f'Two-Stage Framework (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle=':',
             label='Random baseline (AUC = 0.500)')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC on Test Communities (n = {len(df)})')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'roc_comparison.png')
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved ROC plot to {out}")


def plot_feature_importance():
    """
    Feature importance from the coefficients of the trained classifier.

    Signed coefficients are plotted, so the direction of each feature's effect
    is visible. Source: data/model_coefficients.json, written by model.train().
    """
    with open(_require(COEF_JSON), encoding='utf-8') as f:
        payload = json.load(f)

    df = pd.DataFrame({
        'Feature': list(payload['coefficients'].keys()),
        'Coefficient': list(payload['coefficients'].values()),
    })
    df['Abs'] = df['Coefficient'].abs()
    df = df.sort_values('Abs', ascending=True)

    plt.figure(figsize=(10, 6))
    bars = plt.barh(df['Feature'], df['Coefficient'], color='#2ca02c', alpha=0.8)

    for bar in bars:
        width = bar.get_width()
        offset = 0.01 if width >= 0 else -0.01
        plt.text(width + offset, bar.get_y() + bar.get_height() / 2,
                 f'{width:+.3f}', va='center',
                 ha='left' if width >= 0 else 'right', fontsize=10)

    plt.axvline(0, color='black', lw=0.8)
    plt.xlabel('Logistic Regression Coefficient (standardized features)')
    plt.title(f'Feature Importance (best C = {payload["best_C"]:.4g})')
    plt.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'feature_importance.png')
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved Feature Importance plot to {out}")


def plot_community_evolution():
    """
    Community size and internal density across snapshots, grouped by target.

    Bands show the min-max range over communities within each group.
    Source: data/community_evolution.csv (all 10 snapshots x 10 communities).
    """
    df = pd.read_csv(_require(EVOLUTION_CSV))

    panels = [
        ('community_size', 'Community Size (Nodes)', 'Evolution of Community Size'),
        ('internal_density', 'Internal Density', 'Evolution of Internal Density'),
    ]
    groups = [(1, '#d62728', 'Fake-dominant (target = 1)'),
              (0, '#1f77b4', 'Normal (target = 0)')]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (column, ylabel, title) in zip(axes, panels):
        for target, color, name in groups:
            sub = df[df['target'] == target]
            if sub.empty:
                continue
            stats = sub.groupby('snapshot_id')[column].agg(['mean', 'min', 'max'])
            ax.plot(stats.index, stats['mean'], marker='o', color=color, lw=2, label=name)
            ax.fill_between(stats.index, stats['min'], stats['max'], color=color, alpha=0.12)

        ax.set_xlabel('Snapshot ID')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, 'community_evolution.png')
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"Saved Evolution plot to {out}")


if __name__ == "__main__":
    plot_roc_comparison()
    plot_feature_importance()
    plot_community_evolution()
