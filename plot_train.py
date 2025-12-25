import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ================= 配置区域 =================
# 映射文件名到图例名称
FILES = {
    "Proposed (Hybrid GAT)": "train_convergence_Proposed.csv",
    "Baseline (GAT)": "train_convergence_Baseline_GAT.csv",
    "Baseline (GCN)": "train_convergence_Baseline_GCN.csv",
    "No-GNN (Dueling)": "train_convergence_NoGNN_Dueling.csv",
    "No-GNN (Standard)": "train_convergence_NoGNN_Standard.csv"
}

OUTPUT_DIR = "paper_plots_training"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 设置学术绘图风格
sns.set(style="whitegrid", context="paper", font_scale=1.4)
plt.rcParams["font.family"] = "serif"
plt.rcParams["lines.linewidth"] = 2

# ================= 数据加载与平滑 =================
dfs = []
for label, filename in FILES.items():
    if os.path.exists(filename):
        try:
            df = pd.read_csv(filename)
            df['Model'] = label
            # 滑动窗口平滑，让曲线更清晰 (Window size = 20 epochs)
            df['cumulative_reward_smooth'] = df['cumulative_reward'].rolling(window=20).mean()
            df['v2v_success_smooth'] = df['v2v_success_rate'].rolling(window=20).mean()
            df['v2i_capacity_smooth'] = df['v2i_sum_capacity'].rolling(window=20).mean()
            df['loss_smooth'] = df['mean_loss'].rolling(window=20).mean()
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {filename}: {e}")

if not dfs:
    print("未找到任何CSV文件，请确认文件路径。")
    exit()

full_df = pd.concat(dfs)


# ================= 绘图函数 =================
def plot_metric(y_col, title, ylabel, filename, y_limit=None, log_scale=False):
    plt.figure(figsize=(8, 6))

    # 绘制曲线
    sns.lineplot(data=full_df, x="epoch", y=y_col, hue="Model", style="Model",
                 dashes=False, palette="deep")

    plt.title(title, fontsize=16, pad=15)
    plt.xlabel("Training Epochs", fontweight='bold')
    plt.ylabel(ylabel, fontweight='bold')
    plt.legend(loc="best", framealpha=0.9, fontsize=10)

    if y_limit:
        plt.ylim(y_limit)
    if log_scale:
        plt.yscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.close()
    print(f"Saved {filename}")


# ================= 生成图表 =================

# 1. 累积奖励 (最核心的证据：证明综合性能最优)
plot_metric(
    y_col="cumulative_reward_smooth",
    title="Convergence of Cumulative Reward",
    ylabel="Average Reward",
    filename="Training_Reward.png",
    y_limit=(-5, 4)  # 限制Y轴范围，剔除 No-GNN Dueling 的极低值干扰
)

# 2. V2V 成功率 (展示 Proposed 稳定在高位，虽然不是最高)
plot_metric(
    y_col="v2v_success_smooth",
    title="V2V Link Success Rate",
    ylabel="Success Rate",
    filename="Training_V2V_Success.png",
    y_limit=(0.0, 1.05)
)

# 3. V2I 容量 (展示 Proposed 对主用户的保护能力)
plot_metric(
    y_col="v2i_capacity_smooth",
    title="V2I Sum Capacity Protection",
    ylabel="Capacity (Mbps)",
    filename="Training_V2I_Capacity.png"
)

# 4. 训练损失 (展示收敛稳定性)
plot_metric(
    y_col="loss_smooth",
    title="Training Loss Convergence",
    ylabel="Mean Loss (Log Scale)",
    filename="Training_Loss.png",
    log_scale=True  # Loss 差异巨大，必须用对数坐标
)

print(f"\n分析完成！所有图片已保存至 {OUTPUT_DIR}/ 文件夹。")

