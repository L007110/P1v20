import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ================= 配置区域 =================
INPUT_DIR = "paper_results"  # CSV文件所在目录
OUTPUT_DIR = "paper_plots"  # 图片保存目录

# 定义模型列表 (文件名, 图例标签, 颜色, 线型, 标记)
MODELS = [
    {"file": "test_scalability_Proposed.csv", "label": "Proposed (Hybrid GAT)", "color": "#d62728", "fmt": "o-"},
    {"file": "test_scalability_Baseline_GAT.csv", "label": "Baseline (GAT)", "color": "#ff7f0e", "fmt": "s--"},
    {"file": "test_scalability_Baseline_GCN.csv", "label": "Baseline (GCN)", "color": "#2ca02c", "fmt": "^--"},
    {"file": "test_scalability_NoGNN_Dueling.csv", "label": "No-GNN (Dueling)", "color": "#1f77b4", "fmt": "x:"},
    {"file": "test_scalability_NoGNN_Standard.csv", "label": "No-GNN (Standard)", "color": "#9467bd", "fmt": "d:"}
]

# 设置绘图风格
sns.set(style="whitegrid", font_scale=1.4)
plt.rcParams["font.family"] = "serif"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.linestyle"] = "--"


def load_and_clean_data(filepath):
    """
    读取并清洗数据：解决重复写入和乱序问题
    """
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        return None

    try:
        df = pd.read_csv(filepath)

        # 1. 关键步骤：去重
        # keep='last' 表示保留最后一次跑的结果（假设最后一次是最新的）
        df = df.drop_duplicates(subset=['vehicle_count'], keep='last')

        # 2. 关键步骤：排序
        # 必须按车辆数从小到大排，否则线会乱飞
        df = df.sort_values(by='vehicle_count')

        return df
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None


def plot_fig2_success_rate():
    plt.figure(figsize=(8, 6))

    for model in MODELS:
        df = load_and_clean_data(os.path.join(INPUT_DIR, model["file"]))
        if df is not None:
            plt.plot(df['vehicle_count'], df['v2v_success_rate'],
                     model["fmt"], color=model["color"], label=model["label"],
                     linewidth=2.5, markersize=8)

    plt.xlabel("Vehicle Density (Number of Vehicles)")
    plt.ylabel("V2V Link Success Rate")
    plt.title("Scalability: Success Rate vs Density")
    plt.legend(loc="best", fontsize=11, framealpha=0.9)
    plt.ylim(0.4, 1.02)  # 根据你的数据范围调整
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/Fig2_Scalability_V2V_Fixed.png", dpi=300)
    print("Saved Fig2_Scalability_V2V_Fixed.png")


def plot_fig3_delay():
    plt.figure(figsize=(8, 6))

    for model in MODELS:
        df = load_and_clean_data(os.path.join(INPUT_DIR, model["file"]))
        if df is not None:
            # 将延迟显示为对数坐标往往更清晰，或者直接画
            plt.plot(df['vehicle_count'], df['p95_delay_ms'],
                     model["fmt"], color=model["color"], label=model["label"],
                     linewidth=2.5, markersize=8)

    plt.xlabel("Vehicle Density (Number of Vehicles)")
    plt.ylabel("P95 Delay (ms)")
    plt.title("Scalability: Delay Reliability vs Density")
    plt.legend(loc="upper left", fontsize=11, framealpha=0.9)
    plt.yscale('log')  # 延迟通常跨度大，用对数轴更好看
    plt.grid(True, which="minor", ls=":", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/Fig3_Scalability_Delay_Fixed.png", dpi=300)
    print("Saved Fig3_Scalability_Delay_Fixed.png")


def plot_fig4_v2i_capacity():
    plt.figure(figsize=(8, 6))

    for model in MODELS:
        df = load_and_clean_data(os.path.join(INPUT_DIR, model["file"]))
        if df is not None:
            plt.plot(df['vehicle_count'], df['v2i_sum_capacity_mbps'],
                     model["fmt"], color=model["color"], label=model["label"],
                     linewidth=2.5, markersize=8)

    plt.xlabel("Vehicle Density (Number of Vehicles)")
    plt.ylabel("V2I Sum Capacity (Mbps)")
    plt.title("Scalability: V2I Capacity vs Density")
    plt.legend(loc="best", fontsize=11, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/Fig4_Scalability_V2I_Fixed.png", dpi=300)
    print("Saved Fig4_Scalability_V2I_Fixed.png")


if __name__ == "__main__":
    plot_fig2_success_rate()
    plot_fig3_delay()
    plot_fig4_v2i_capacity()