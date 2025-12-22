import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# ==========================================
# 绘图配置
# ==========================================
RESULTS_DIR = "paper_results"
OUTPUT_DIR = "paper_plots"

# 定义模型样式 (颜色和线型)
MODELS = [
    {"id": "Proposed", "label": "Proposed (Hybrid GAT)", "color": "#d62728", "style": "-"},  # 红色 (重点)
    {"id": "Baseline_GAT", "label": "GAT (No Hybrid)", "color": "#ff7f0e", "style": "--"},  # 橙色
    {"id": "Baseline_GCN", "label": "GCN (Classic)", "color": "#2ca02c", "style": "--"},  # 绿色
    {"id": "NoGNN_Dueling", "label": "Dueling DQN", "color": "#1f77b4", "style": ":"},  # 蓝色
    {"id": "NoGNN_Standard", "label": "Standard DQN", "color": "#7f7f7f", "style": ":"}  # 灰色
]


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def smooth(data, window=20):
    """滑动平均平滑函数，让曲线更美观"""
    return data.rolling(window=window, min_periods=1).mean()


def plot_convergence():
    """图1：收敛性分析 (Reward vs Epoch)"""
    print("Generating Figure 1: Convergence Analysis...")
    plt.figure(figsize=(8, 6))
    sns.set(style="whitegrid", font_scale=1.1)

    for model in MODELS:
        file_path = f"{RESULTS_DIR}/train_convergence_{model['id']}.csv"
        if not os.path.exists(file_path):
            continue

        df = pd.read_csv(file_path)

        x = df['epoch']
        y = smooth(df['cumulative_reward'])

        # 绘制主线
        plt.plot(x, y, label=model['label'], color=model['color'],
                 linestyle=model['style'], linewidth=2.5)

        # 绘制阴影 (Raw Data) 以展示波动范围
        plt.plot(x, df['cumulative_reward'], color=model['color'], alpha=0.1, linewidth=0.5)

    plt.xlabel("Training Epochs", fontweight='bold')
    plt.ylabel("Cumulative Reward", fontweight='bold')
    plt.title("Convergence Performance (Density N=60)", fontweight='bold')
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/Fig1_Convergence.png", dpi=300)
    plt.close()


def plot_scalability_v2v():
    """图2：可扩展性分析 - V2V成功率 (这是最重要的图)"""
    print("Generating Figure 2: Scalability (V2V Success Rate)...")
    plt.figure(figsize=(8, 6))
    sns.set(style="whitegrid", font_scale=1.1)

    markers = ['o', 's', '^', 'D', 'x']

    for i, model in enumerate(MODELS):
        file_path = f"{RESULTS_DIR}/test_scalability_{model['id']}.csv"
        if not os.path.exists(file_path):
            continue

        df = pd.read_csv(file_path)

        # 绘制曲线
        plt.plot(df['vehicle_count'], df['v2v_success_rate'] * 100,  # 转换为百分比
                 label=model['label'], color=model['color'],
                 marker=markers[i], markersize=8, linewidth=2.5, linestyle=model['style'])

    plt.xlabel("Vehicle Density (Number of Vehicles)", fontweight='bold')
    plt.ylabel("V2V Success Rate (%)", fontweight='bold')
    plt.title("Scalability & Generalization Analysis", fontweight='bold')
    plt.ylim(0, 105)
    plt.legend(loc="lower left", frameon=True, fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/Fig2_Scalability_V2V.png", dpi=300)
    plt.close()


def plot_scalability_delay():
    """图3：可扩展性分析 - 延迟 (P95)"""
    print("Generating Figure 3: Scalability (Latency)...")
    plt.figure(figsize=(8, 6))
    sns.set(style="whitegrid", font_scale=1.1)

    markers = ['o', 's', '^', 'D', 'x']

    for i, model in enumerate(MODELS):
        file_path = f"{RESULTS_DIR}/test_scalability_{model['id']}.csv"
        if not os.path.exists(file_path):
            continue

        df = pd.read_csv(file_path)

        plt.plot(df['vehicle_count'], df['p95_delay_ms'],
                 label=model['label'], color=model['color'],
                 marker=markers[i], markersize=8, linewidth=2.5, linestyle=model['style'])

    plt.xlabel("Vehicle Density (Number of Vehicles)", fontweight='bold')
    plt.ylabel("P95 Latency (ms)", fontweight='bold')
    plt.title("Latency Degradation vs. Density", fontweight='bold')
    plt.legend(loc="upper left", frameon=True, fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/Fig3_Scalability_Delay.png", dpi=300)
    plt.close()


def plot_scalability_v2i():
    """图4：可扩展性分析 - V2I 容量 (约束满足性证明)"""
    print("Generating Figure 4: Scalability (V2I Capacity)...")
    plt.figure(figsize=(8, 6))
    sns.set(style="whitegrid", font_scale=1.1)

    markers = ['o', 's', '^', 'D', 'x']

    # 1. 绘制各模型的曲线
    for i, model in enumerate(MODELS):
        file_path = f"{RESULTS_DIR}/test_scalability_{model['id']}.csv"
        if not os.path.exists(file_path):
            continue

        df = pd.read_csv(file_path)

        # 你的 CSV 中列名应该是 'v2i_sum_capacity_mbps'
        plt.plot(df['vehicle_count'], df['v2i_sum_capacity_mbps'],
                 label=model['label'], color=model['color'],
                 marker=markers[i], markersize=8, linewidth=2.5, linestyle=model['style'])

    # 2. 绘制阈值线 (Constraint Threshold)
    # 这条红色的虚线非常重要，它代表了"及格线"
    # 根据 Parameters.py，默认 V2I 阈值乘数影响下，假设阈值为某定值(比如 1.0 或 2.0)
    # 这里我们画一条示意线，或者不画也可以，但画出来更有说服力
    # plt.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='QoS Threshold')

    plt.xlabel("Vehicle Density (Number of Vehicles)", fontweight='bold')
    plt.ylabel("V2I Sum Capacity (Mbps)", fontweight='bold')
    plt.title("V2I Capacity & Constraint Satisfaction", fontweight='bold')
    plt.legend(loc="best", frameon=True, fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/Fig4_Scalability_V2I.png", dpi=300)
    plt.close()


def main():
    ensure_dir(OUTPUT_DIR)

    # 生成四张图
    plot_convergence()  # 图1
    plot_scalability_v2v()  # 图2
    plot_scalability_delay()  # 图3
    plot_scalability_v2i()  # 图4

    print(f"\n所有图表已生成，保存在 '{OUTPUT_DIR}/' 目录中。")


if __name__ == "__main__":
    main()