import os
import sys
import time

# ==========================================
# 论文核心消融实验脚本: GNN vs No-GNN (自动化版)
# ==========================================

# 1. 实验配置
# 使用我们在 Grid Search 中找到的最佳参数
BEST_SNR = 0.5
BEST_V2I = 1.0
EPOCHS = 1000  # 建议跑久一点，观察长期收敛性
SEED = 42


def run_ablation():
    py_cmd = f'"{sys.executable}"'

    print("==================================================")
    print("   STARTING ABLATION STUDY: PROPOSED vs BASELINE  ")
    print("==================================================")

    # -------------------------------------------------------
    # 任务 1: Proposed Method (GNN-Hybrid)
    # -------------------------------------------------------
    print(f"\n>>> [1/2] Running Proposed Method (GNN-Hybrid) <<<")
    cmd_gnn = (
        f"{py_cmd} Main.py --run_mode TRAIN "
        f"--seed {SEED} --epochs {EPOCHS} "
        f"--use_gnn True "  # <--- 开启 GNN
        f"--gnn_arch HYBRID "
        f"--snr_mul {BEST_SNR} --v2i_mul {BEST_V2I}"
    )
    exit_code = os.system(cmd_gnn)

    if exit_code == 0 and os.path.exists("training_results/global_metrics.csv"):
        os.rename("training_results/global_metrics.csv", "training_results/ablation_GNN_Hybrid.csv")
        print(">>> [SUCCESS] Proposed method results saved to 'training_results/ablation_GNN_Hybrid.csv'")
    else:
        print(">>> [FAIL] Proposed method run failed!")

    # -------------------------------------------------------
    # 任务 2: Baseline Method (No-GNN / Pure Dueling DQN)
    # -------------------------------------------------------
    print(f"\n>>> [2/2] Running Baseline (No-GNN) <<<")
    cmd_baseline = (
        f"{py_cmd} Main.py --run_mode TRAIN "
        f"--seed {SEED} --epochs {EPOCHS} "
        f"--use_gnn False "  # <--- 关闭 GNN
        f"--snr_mul {BEST_SNR} --v2i_mul {BEST_V2I}"
    )
    exit_code = os.system(cmd_baseline)

    if exit_code == 0 and os.path.exists("training_results/global_metrics.csv"):
        os.rename("training_results/global_metrics.csv", "training_results/ablation_NoGNN.csv")
        print(">>> [SUCCESS] Baseline results saved to 'training_results/ablation_NoGNN.csv'")
    else:
        print(">>> [FAIL] Baseline run failed!")

    print("\n==================================================")
    print("   ABLATION STUDY COMPLETE.   ")
    print("==================================================")


if __name__ == "__main__":
    run_ablation()