import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import numpy as np

# ================= Configuration Area =================
# Directory containing your CSV files (use "." if in current directory)
INPUT_DIR = "paper_results"
OUTPUT_DIR = "paper_plots"

# [CRITICAL PARAMETER] Weight factor lambda for the combined score
# Lambda = 1.0 means: 100% V2V Success Rate is weighted equally to 1000 Mbps V2I Capacity
# Adjust this if you want to emphasize V2I protection (e.g., 1.2) or V2V reliability (e.g., 0.8)
LAMBDA_VAL = 1.0

# Define model list: includes filename, legend label, color, line style, and line width
MODELS = [
    {"file": "test_scalability_Proposed.csv", "label": "Proposed (Hybrid GAT)", "color": "#d62728", "fmt": "o-",
     "lw": 3},
    {"file": "test_scalability_Baseline_GAT.csv", "label": "Baseline (GAT)", "color": "#ff7f0e", "fmt": "s--", "lw": 2},
    {"file": "test_scalability_Baseline_GCN.csv", "label": "Baseline (GCN)", "color": "#2ca02c", "fmt": "^--", "lw": 2},
    {"file": "test_scalability_NoGNN_Dueling.csv", "label": "No-GNN (Dueling)", "color": "#1f77b4", "fmt": "x:",
     "lw": 2},
    {"file": "test_scalability_NoGNN_Standard.csv", "label": "No-GNN (Standard)", "color": "#9467bd", "fmt": "d:",
     "lw": 2}
]

# Set academic plotting style
sns.set(style="ticks", context="paper", font_scale=1.4)
plt.rcParams["font.family"] = "serif"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.alpha"] = 0.6


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def load_data():
    dfs = []
    for model in MODELS:
        filepath = os.path.join(INPUT_DIR, model["file"])
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                # Basic cleaning
                df = df.sort_values(by='vehicle_count')
                df = df.drop_duplicates(subset=['vehicle_count'], keep='last')

                # Add model metadata
                df['Model'] = model["label"]
                df['Color'] = model["color"]
                df['Fmt'] = model["fmt"]
                df['Lw'] = model["lw"]

                # ==========================================
                # [NEW METRIC] Calculate Effective Throughput Score
                # Score = V2V_Success_Rate + lambda * (V2I_Capacity / 1000)
                # ==========================================
                if 'v2i_sum_capacity_mbps' in df.columns and 'v2v_success_rate' in df.columns:
                    df['effective_score'] = df['v2v_success_rate'] + LAMBDA_VAL * (df['v2i_sum_capacity_mbps'] / 1000.0)
                else:
                    print(f"Warning: Missing columns for score calculation in {model['file']}")
                    df['effective_score'] = np.nan

                dfs.append(df)
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
    if not dfs:
        return None
    return pd.concat(dfs)


def print_quantitative_analysis(full_df):
    """Print quantitative analysis for paper writing"""
    print("\n" + "=" * 50)
    print("   QUANTITATIVE ANALYSIS (For Paper)   ")
    print("=" * 50)

    # Analyze at highest density N=120 (most challenging scenario)
    target_density = 120
    subset = full_df[full_df['vehicle_count'] == target_density]

    if subset.empty:
        print("No data for N=120.")
        return

    # Get Proposed data
    proposed_row = subset[subset['Model'] == "Proposed (Hybrid GAT)"]
    if proposed_row.empty:
        return

    prop_v2v = proposed_row['v2v_success_rate'].values[0]
    prop_v2i = proposed_row['v2i_sum_capacity_mbps'].values[0]
    prop_score = proposed_row['effective_score'].values[0]

    print(f"Metrics at Density N={target_density}:")
    print(f"  Proposed -> V2V: {prop_v2v:.1%}, V2I: {prop_v2i:.1f} Mbps")
    print(f"  Proposed -> Effective Score: {prop_score:.4f}")
    print("-" * 30)

    # Compare with Baseline (GAT)
    gat_row = subset[subset['Model'] == "Baseline (GAT)"]
    if not gat_row.empty:
        gat_v2v = gat_row['v2v_success_rate'].values[0]
        gat_v2i = gat_row['v2i_sum_capacity_mbps'].values[0]
        gat_score = gat_row['effective_score'].values[0]

        score_diff = (prop_score - gat_score) / gat_score * 100

        print(f"Vs Baseline (GAT):")
        print(f"  V2V Gap: {prop_v2v - gat_v2v:+.1%} (Trade-off)")
        print(f"  V2I Gain: {(prop_v2i - gat_v2i) / gat_v2i * 100:+.1f}%")
        print(f"  Score Gain: {score_diff:+.2f}% (Overall Improvement)")

    print("-" * 30)

    # Compare with No-GNN (Standard)
    std_row = subset[subset['Model'] == "No-GNN (Standard)"]
    if not std_row.empty:
        std_score = std_row['effective_score'].values[0]
        score_diff = (prop_score - std_score) / std_score * 100

        print(f"Vs No-GNN (Standard):")
        print(f"  Score Gain: {score_diff:+.2f}%")
        print("  Result: Proposed demonstrates superior overall system utility.")


def plot_metrics(full_df):
    ensure_dir(OUTPUT_DIR)

    # 1. V2V Success Rate
    plt.figure(figsize=(8, 6))
    for model in MODELS:
        data = full_df[full_df['Model'] == model["label"]]
        if not data.empty:
            plt.plot(data['vehicle_count'], data['v2v_success_rate'],
                     model["fmt"], color=model["color"], label=model["label"],
                     linewidth=model["lw"], markersize=8)

    plt.xlabel("Vehicle Density (Number of Vehicles)", fontweight='bold')
    plt.ylabel("V2V Link Success Rate", fontweight='bold')
    plt.title("Reliability: V2V Success Rate", fontsize=14, y=1.02)
    plt.ylim(0.45, 1.02)  # Adjust view to highlight high-end differences
    plt.legend(loc="lower left", fontsize=10, framealpha=0.95)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/Fig_Scalability_V2V.png", dpi=300)
    plt.close()

    # 2. V2I Capacity
    plt.figure(figsize=(8, 6))
    for model in MODELS:
        data = full_df[full_df['Model'] == model["label"]]
        if not data.empty:
            plt.plot(data['vehicle_count'], data['v2i_sum_capacity_mbps'],
                     model["fmt"], color=model["color"], label=model["label"],
                     linewidth=model["lw"], markersize=8)

    plt.xlabel("Vehicle Density (Number of Vehicles)", fontweight='bold')
    plt.ylabel("V2I Sum Capacity (Mbps)", fontweight='bold')
    plt.title("Interference Management: V2I Capacity", fontsize=14, y=1.02)
    plt.legend(loc="best", fontsize=10, framealpha=0.95)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/Fig_Scalability_V2I.png", dpi=300)
    plt.close()

    # 3. Delay (Log Scale)
    plt.figure(figsize=(8, 6))
    for model in MODELS:
        data = full_df[full_df['Model'] == model["label"]]
        if not data.empty:
            plt.plot(data['vehicle_count'], data['p95_delay_ms'],
                     model["fmt"], color=model["color"], label=model["label"],
                     linewidth=model["lw"], markersize=8)

    plt.xlabel("Vehicle Density (Number of Vehicles)", fontweight='bold')
    plt.ylabel("95th Percentile Latency (ms)", fontweight='bold')
    plt.yscale('log')
    plt.title("Latency: P95 Delay", fontsize=14, y=1.02)
    plt.legend(loc="upper left", fontsize=10, framealpha=0.95)
    plt.grid(True, which="minor", ls=":", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/Fig_Scalability_Delay.png", dpi=300)
    plt.close()

    # 4. [NEW] Effective Throughput Score
    plt.figure(figsize=(8, 6))

    # Dynamic Y-axis limits
    y_vals = full_df['effective_score'].dropna()
    y_min, y_max = y_vals.min(), y_vals.max()

    for model in MODELS:
        data = full_df[full_df['Model'] == model["label"]]
        if not data.empty:
            plt.plot(data['vehicle_count'], data['effective_score'],
                     model["fmt"], color=model["color"], label=model["label"],
                     linewidth=model["lw"], markersize=9)  # Slightly larger markers for the main result

    plt.xlabel("Vehicle Density (Number of Vehicles)", fontweight='bold')
    plt.ylabel(f"Effective Score ($\lambda$={LAMBDA_VAL})", fontweight='bold')
    plt.title("System Effectiveness: Weighted Performance", fontsize=14, y=1.02)
    plt.legend(loc="best", fontsize=10, framealpha=0.95)
    plt.ylim(y_min * 0.98, y_max * 1.02)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/Fig_Scalability_Effective_Score.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    df = load_data()
    if df is not None:
        plot_metrics(df)
        print_quantitative_analysis(df)
        print(f"\nPlots saved to {OUTPUT_DIR}/")
        print(f"Check '{OUTPUT_DIR}/Fig_Scalability_Effective_Score.png' for the combined metric.")
    else:
        print("No data found. Please check CSV file paths.")