import os
import sys
import shutil
import time


# ==========================================
# 实验配置 (CONFIGURATION)
# ==========================================
PYTHON_EXEC = sys.executable  # 获取当前Python解释器路径
TRAIN_EPOCHS = 1000  #
TRAIN_VEHICLES = 60  # 训练时的车辆密度 (作为泛化能力的基准)
SEED = 11  # 随机种子
RESULTS_DIR = "paper_results"  # 结果保存目录

# 定义 5 个对比模型
MODELS = [
    {
        "id": "Proposed",
        "label": "Proposed (Hybrid GAT)",
        "gnn": "True", "arch": "HYBRID", "dueling": "True"
    },
    {
        "id": "Baseline_GAT",
        "label": "Baseline (GAT)",
        "gnn": "True", "arch": "GAT", "dueling": "False"
    },
    {
        "id": "Baseline_GCN",
        "label": "Baseline (GCN)",
        "gnn": "True", "arch": "GCN", "dueling": "False"
    },
    {
        "id": "NoGNN_Dueling",
        "label": "No-GNN (Dueling DQN)",
        "gnn": "False", "arch": "HYBRID", "dueling": "True"
    },
    {
        "id": "NoGNN_Standard",
        "label": "No-GNN (Standard DQN)",
        "gnn": "False", "arch": "HYBRID", "dueling": "False"
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

    # =========================================================================
    # 第一阶段: 训练 (收敛性分析)
    # 在固定密度 (N=60) 下训练所有模型
    # =========================================================================
    print("\n" + "=" * 50)
    print("PHASE 1: TRAINING (Convergence Analysis)")
    print("=" * 50)

    for model in MODELS:
        print(f"\n>>> Training Model: {model['label']} <<<")

        # 构造训练命令
        cmd = (
            f"{PYTHON_EXEC} Main.py --run_mode TRAIN "
            f"--epochs {TRAIN_EPOCHS} "
            f"--seed {SEED} "
            f"--vehicle_count {TRAIN_VEHICLES} "
            f"--use_gnn {model['gnn']} "
            f"--gnn_arch {model['arch']} "
            f"--dueling {model['dueling']} "
        )
        run_command(cmd)

        # 移动/重命名结果文件
        src_csv = "training_results/global_metrics.csv"
        dst_csv = f"{RESULTS_DIR}/train_convergence_{model['id']}.csv"

        # 确定生成的模型文件名 (Main.py 中的逻辑)
        if model['gnn'] == "True":
            src_model = f"model_{model['arch']}.pt"
        else:
            src_model = "model_NoGNN_Baseline_v2.pth" if model['dueling'] == "True" else "model_Standard_DQN.pth"

        dst_model = f"{RESULTS_DIR}/model_{model['id']}.pth"

        # 保存 CSV
        if os.path.exists(src_csv):
            shutil.copy(src_csv, dst_csv)
            print(f"Saved training log to: {dst_csv}")

        # 保存模型权重 (供测试阶段使用)
        if os.path.exists(src_model):
            shutil.copy(src_model, dst_model)
            print(f"Saved model checkpoint to: {dst_model}")
        else:
            print(f"Warning: Model file {src_model} not found.")

    # =========================================================================
    # 第二阶段: 可扩展性测试 (密度分析)
    # 加载训练好的模型，在 N=[20, 40, 60, 80, 100, 120] 上进行测试
    # =========================================================================
    print("\n" + "=" * 50)
    print("PHASE 2: SCALABILITY TESTING (Generalization)")
    print("=" * 50)

    for model in MODELS:
        print(f"\n>>> Testing Scalability: {model['label']} <<<")

        saved_model_path = f"{RESULTS_DIR}/model_{model['id']}.pth"

        if not os.path.exists(saved_model_path):
            print(f"Skipping {model['id']} - Checkpoint not found.")
            continue

        # 【关键步骤】将保存的模型复制回 Main.py 期望的文件名
        if model['gnn'] == "True":
            target_name = f"model_{model['arch']}.pt"
        else:
            target_name = "model_NoGNN_Baseline_v2.pth" if model['dueling'] == "True" else "model_Standard_DQN.pth"

        shutil.copy(saved_model_path, target_name)

        # 运行测试模式 (Main.py 会自动遍历不同的车辆密度)
        cmd = (
            f"{PYTHON_EXEC} Main.py --run_mode TEST "
            f"--use_gnn {model['gnn']} "
            f"--gnn_arch {model['arch']} "
            f"--dueling {model['dueling']} "
        )
        run_command(cmd)

        # 移动测试结果 CSV
        # Main.py 生成的文件名通常包含后缀，我们需要通配或者构建正确的名字
        # 根据 Parameters.py 的逻辑，后缀是 _VehDef_{ARCH}

        src_csv = f"training_results/scalability_VehDef_{model['arch']}.csv"
        dst_csv = f"{RESULTS_DIR}/test_scalability_{model['id']}.csv"

        if os.path.exists(src_csv):
            shutil.copy(src_csv, dst_csv)
            print(f"Saved scalability log to: {dst_csv}")
        else:
            print(f"Warning: Scalability CSV {src_csv} not found. Check Main.py output.")

    print("\n所有实验已完成。结果保存在 'paper_results/' 目录中。")


if __name__ == "__main__":
    main()