import os
import sys
import shutil

# ==========================================
# CONFIGURATION
# ==========================================
PYTHON_EXEC = sys.executable
TRAIN_EPOCHS = 800  # 建议跑足够长，观察收敛后的物理指标
SEED = 11  # 固定种子，控制变量
RESULTS_DIR = "paper_results/reward_ablation"

# 最佳基础参数 (来自之前的 Grid Search 或经验值)
BASE_SNR = 0.5
BASE_V2I = 1.0
BASE_DELAY = 1.0
BASE_POWER = 1.0

# 这是一个消融列表：每次把某一项设为 0
ABLATION_VARIANTS = [
    {
        "id": "Full_Reward",
        "label": "Full Reward (Proposed)",
        "params": f"--snr_mul {BASE_SNR} --v2i_mul {BASE_V2I} --delay_mul {BASE_DELAY} --power_mul {BASE_POWER}"
    },
    {
        "id": "No_V2I",
        "label": "w/o V2I Constraint",
        "params": f"--snr_mul {BASE_SNR} --v2i_mul 0.0 --delay_mul {BASE_DELAY} --power_mul {BASE_POWER}"
    },
    {
        "id": "No_Delay",
        "label": "w/o Delay Penalty",
        "params": f"--snr_mul {BASE_SNR} --v2i_mul {BASE_V2I} --delay_mul 0.0 --power_mul {BASE_POWER}"
    },
    {
        "id": "No_Power",
        "label": "w/o Power Efficiency",
        "params": f"--snr_mul {BASE_SNR} --v2i_mul {BASE_V2I} --delay_mul {BASE_DELAY} --power_mul 0.0"
    }
]


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def run_command(cmd):
    print(f"Executing: {cmd}")
    exit_code = os.system(cmd)
    if exit_code != 0:
        print(f"!!! Error executing command. Exit code: {exit_code}")


def main():
    ensure_dir(RESULTS_DIR)

    print("==================================================")
    print("   STARTING REWARD ABLATION STUDY  ")
    print("==================================================")

    for variant in ABLATION_VARIANTS:
        print(f"\n>>> Running Variant: {variant['label']} <<<")

        # 我们使用 GNN-Hybrid 作为固定架构，只改变奖励参数
        cmd = (
            f"{PYTHON_EXEC} Main.py --run_mode TRAIN "
            f"--epochs {TRAIN_EPOCHS} "
            f"--seed {SEED} "
            f"--use_gnn True "
            f"--gnn_arch HYBRID "
            f"{variant['params']}"
        )
        run_command(cmd)

        # 移动结果 CSV
        src_csv = "training_results/global_metrics.csv"
        dst_csv = f"{RESULTS_DIR}/metrics_{variant['id']}.csv"

        if os.path.exists(src_csv):
            shutil.copy(src_csv, dst_csv)
            print(f">>> Saved logs to: {dst_csv}")
        else:
            print(">>> [FAIL] Log file not found!")

    print("\nReward ablation complete.")
    print(f"Results stored in '{RESULTS_DIR}/'")


if __name__ == "__main__":
    main()