import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ================= 配置区域 =================
# CSV 文件路径 (请确保这两个文件存在)
FILE_GNN = "training_results/ablation_GNN_Hybrid.csv"
FILE_NOGNN = "training_results/ablation_NoGNN.csv"

# 输出图片路径
OUTPUT_IMG = "training_results/ablation_comparison.png"

# 平滑窗口大小 (使曲线更平滑好看，符合论文标准)
WINDOW_SIZE = 20


# ===========================================

def plot_ablation():
    # 1. 检查文件是否存在
    if not os.path.exists(FILE_GNN) or not os.path.exists(FILE_NOGNN):
        print(f"[Error] 找不到 CSV 文件！请确认 {FILE_GNN} 和 {FILE_NOGNN} 是否存在。")
        return

    # 2. 读取数据
    df_gnn = pd.read_csv(FILE_GNN)
    df_nognn = pd.read_csv(FILE_NOGNN)

    print(f"Loaded GNN data: {len(df_gnn)} epochs")
    print(f"Loaded No-GNN data: {len(df_nognn)} epochs")

    # 3. 设置绘图风格 (Seaborn 论文风格)
    sns.set(style="whitegrid", font_scale=1.2)
    plt.rcParams["font.family"] = "DejaVu Sans"

    # 创建画布 (3行1列)
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

    # 定义要画的指标和标签
    metrics = [
        {"col": "cumulative_reward", "label": "Average Reward", "title": "(a) Convergence of Reward"},
        {"col": "v2v_success_rate", "label": "V2V Success Rate", "title": "(b) V2V Reliability Analysis"},
        {"col": "v2i_sum_capacity", "label": "V2I Sum Capacity (Mbps)", "title": "(c) V2I Constraint Satisfaction"}
    ]

    # 4. 开始循环绘图
    for i, metric in enumerate(metrics):
        ax = axes[i]
        col = metric["col"]

        # 检查列名是否存在
        if col not in df_gnn.columns:
            print(f"[Warning] Column '{col}' not found in CSV. Skipping...")
            continue

        # 获取数据并进行平滑处理 (Rolling Mean)
        # GNN (Proposed) - 通常用红色或深色线条
        data_gnn = df_gnn[col].rolling(window=WINDOW_SIZE, min_periods=1).mean()
        # No-GNN (Baseline) - 通常用灰色或虚线
        data_nognn = df_nognn[col].rolling(window=WINDOW_SIZE, min_periods=1).mean()

        # 绘制曲线
        # Proposed Method
        ax.plot(df_gnn['epoch'], data_gnn,
                label="Proposed (GNN-Hybrid)", color="#d62728", linewidth=2.5)

        # Baseline
        ax.plot(df_nognn['epoch'], data_nognn,
                label="Baseline (No-GNN)", color="#7f7f7f", linewidth=2.0, linestyle="--")

        # 绘制阴影带 (表示原始数据的波动范围/方差，这里简化用 raw data 的半透明线代替标准差)
        # 真正的标准差需要多次实验，这里我们画淡色的原始数据作为背景
        ax.plot(df_gnn['epoch'], df_gnn[col], color="#d62728", alpha=0.15, linewidth=1)
        ax.plot(df_nognn['epoch'], df_nognn[col], color="#7f7f7f", alpha=0.15, linewidth=1)

        # 设置图表细节
        ax.set_ylabel(metric["label"], fontweight='bold')
        ax.set_title(metric["title"], loc='left', fontweight='bold', fontsize=14)
        ax.legend(loc="best", frameon=True)
        ax.grid(True, linestyle=':', alpha=0.6)

        # 特殊处理：如果是 V2V 成功率，转换成百分比显示
        if col == "v2v_success_rate":
            vals = ax.get_yticks()
            ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])

    # 设置横轴
    axes[-1].set_xlabel("Training Epochs", fontweight='bold')
    axes[-1].set_xlim(0, max(len(df_gnn), len(df_nognn)))

    plt.tight_layout()
    plt.savefig(OUTPUT_IMG, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\n[Success] Comparison plot saved to: {OUTPUT_IMG}")
    print("Go check it! This is your key evidence.")


if __name__ == "__main__":
    plot_ablation()