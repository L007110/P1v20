import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# ================= 配置区域 (Configuration) =================
INPUT_DIR = "paper_results"  # 输入CSV文件所在的文件夹
OUTPUT_DIR = "paper_plots"  # 输出图片保存的文件夹

# 定义模型列表：包含文件名、图例标签、颜色、线型和标记点
MODELS = [
    {
        "file": "test_scalability_Proposed.csv",
        "label": "Proposed (Hybrid GAT)",
        "color": "#d62728",  # 红色
        "fmt": "o-"  # 圆点实线
    },
    {
        "file": "test_scalability_Baseline_GAT.csv",
        "label": "Baseline (GAT)",
        "color": "#ff7f0e",  # 橙色
        "fmt": "s--"  # 方块虚线
    },
    {
        "file": "test_scalability_Baseline_GCN.csv",
        "label": "Baseline (GCN)",
        "color": "#2ca02c",  # 绿色
        "fmt": "^--"  # 三角虚线
    },
    {
        "file": "test_scalability_NoGNN_Dueling.csv",
        "label": "No-GNN (Dueling)",
        "color": "#1f77b4",  # 蓝色
        "fmt": "x:"  # 叉号点线
    },
    {
        "file": "test_scalability_NoGNN_Standard.csv",
        "label": "No-GNN (Standard)",
        "color": "#9467bd",  # 紫色
        "fmt": "d:"  # 菱形点线
    }
]

# ================= 绘图风格设置 (Style Settings) =================
# 使用 Seaborn 设置学术风格
sns.set(style="whitegrid", context="paper", font_scale=1.5)
plt.rcParams["font.family"] = "serif"  # 衬线字体 (类似 Times New Roman)
plt.rcParams["axes.linewidth"] = 1.5  # 坐标轴线宽
plt.rcParams["lines.linewidth"] = 2.5  # 曲线线宽
plt.rcParams["lines.markersize"] = 9  # 标记点大小
plt.rcParams["axes.grid"] = True  # 开启网格
plt.rcParams["grid.linestyle"] = "--"  # 网格虚线
plt.rcParams["grid.alpha"] = 0.6  # 网格透明度


# ================= 工具函数 (Helper Functions) =================

def ensure_dir(directory):
    """确保输出目录存在"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")


def load_and_clean_data(filepath):
    """
    读取并清洗数据：
    1. 检查文件是否存在
    2. 按照车辆密度排序 (防止连线混乱)
    3. 去除重复的车辆密度记录 (保留最新的)
    """
    if not os.path.exists(filepath):
        print(f"[Warning] File not found: {filepath}")
        return None

    try:
        df = pd.read_csv(filepath)

        # 检查必要的列
        required_cols = ['vehicle_count']
        if not all(col in df.columns for col in required_cols):
            print(f"[Error] Missing columns in {filepath}")
            return None

        # 排序：确保X轴从小到大
        df = df.sort_values(by='vehicle_count')

        # 去重：如果同一个密度跑了多次，保留最后一次的结果
        df = df.drop_duplicates(subset=['vehicle_count'], keep='last')

        return df
    except Exception as e:
        print(f"[Error] Failed to read {filepath}: {e}")
        return None


# ================= 绘图函数 (Plotting Functions) =================

def plot_success_rate():
    """绘制 V2V 链路成功率对比图"""
    print("Plotting V2V Success Rate...")
    plt.figure(figsize=(10, 7))

    for model in MODELS:
        filepath = os.path.join(INPUT_DIR, model["file"])
        df = load_and_clean_data(filepath)

        if df is not None and 'v2v_success_rate' in df.columns:
            plt.plot(df['vehicle_count'], df['v2v_success_rate'],
                     model["fmt"], color=model["color"], label=model["label"])

    plt.xlabel("Vehicle Density (Number of Vehicles)", fontweight='bold')
    plt.ylabel("V2V Link Success Rate", fontweight='bold')
    plt.title("Reliability Analysis: Success Rate vs Density", fontsize=16, y=1.02)
    plt.legend(loc="lower left", frameon=True, framealpha=0.95, edgecolor='gray')
    plt.ylim(0.0, 1.05)  # 成功率范围固定在 0-1

    output_path = os.path.join(OUTPUT_DIR, "Fig2_Scalability_SuccessRate.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_delay():
    """绘制 P95 延迟对比图 (使用对数坐标)"""
    print("Plotting P95 Delay...")
    plt.figure(figsize=(10, 7))

    for model in MODELS:
        filepath = os.path.join(INPUT_DIR, model["file"])
        df = load_and_clean_data(filepath)

        if df is not None and 'p95_delay_ms' in df.columns:
            plt.plot(df['vehicle_count'], df['p95_delay_ms'],
                     model["fmt"], color=model["color"], label=model["label"])

    plt.xlabel("Vehicle Density (Number of Vehicles)", fontweight='bold')
    plt.ylabel("95th Percentile Latency (ms)", fontweight='bold')
    plt.title("Latency Analysis: P95 Delay vs Density", fontsize=16, y=1.02)

    # 延迟通常使用对数坐标展示差异
    plt.yscale('log')
    plt.grid(True, which="minor", ls=":", alpha=0.4)  # 添加次级网格

    plt.legend(loc="upper left", frameon=True, framealpha=0.95, edgecolor='gray')

    output_path = os.path.join(OUTPUT_DIR, "Fig3_Scalability_Delay.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_v2i_capacity():
    """绘制 V2I 总容量对比图"""
    print("Plotting V2I Capacity...")
    plt.figure(figsize=(10, 7))

    for model in MODELS:
        filepath = os.path.join(INPUT_DIR, model["file"])
        df = load_and_clean_data(filepath)

        if df is not None and 'v2i_sum_capacity_mbps' in df.columns:
            plt.plot(df['vehicle_count'], df['v2i_sum_capacity_mbps'],
                     model["fmt"], color=model["color"], label=model["label"])

    plt.xlabel("Vehicle Density (Number of Vehicles)", fontweight='bold')
    plt.ylabel("V2I Sum Capacity (Mbps)", fontweight='bold')
    plt.title("Throughput Analysis: V2I Capacity vs Density", fontsize=16, y=1.02)
    plt.legend(loc="best", frameon=True, framealpha=0.95, edgecolor='gray')

    output_path = os.path.join(OUTPUT_DIR, "Fig4_Scalability_V2I_Capacity.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# ================= 主程序入口 (Main Execution) =================

if __name__ == "__main__":
    # 1. 准备环境
    ensure_dir(OUTPUT_DIR)

    # 2. 检查输入目录
    if not os.path.exists(INPUT_DIR):
        print(f"[Error] Input directory '{INPUT_DIR}' does not exist!")
        print("Please run 'run_paper_experiments.py' first to generate results.")
        sys.exit(1)

    print("=========================================")
    print("   Generating Comparison Plots...        ")
    print("=========================================")

    # 3. 生成三个核心图表
    plot_success_rate()
    plot_delay()
    plot_v2i_capacity()

    print("\nAll plots generated successfully!")
    print(f"Check the '{OUTPUT_DIR}' directory.")