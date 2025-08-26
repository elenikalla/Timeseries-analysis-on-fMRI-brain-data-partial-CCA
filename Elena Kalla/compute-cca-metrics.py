import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# === Settings ===
SUBJECT_IDS = [
    "sub-0001", "sub-0002", "sub-0003", "sub-0004", "sub-0005",
    "sub-0006", "sub-0007", "sub-0008", "sub-0009", "sub-0011"
]
CONDITIONS = ["task", "rest"]
N_ROIS = 100
INPUT_DIR = "preprocessed data/correlation_csvs"
METRIC_OUTPUT_DIR = "preprocessed data/cca_results/metrics"
FIGURE_OUTPUT_DIR = "preprocessed data/cca_results/figures"

os.makedirs(METRIC_OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_OUTPUT_DIR, exist_ok=True)

# === Load CCA matrix from CSV ===
def load_cca_matrix_from_csv(path, n_rois=100):
    df = pd.read_csv(path, header=None)

    # Remove first column if it's an index
    if df.shape[1] > n_rois:
        df = df.iloc[:, 1:]

    # Remove first row if it's also labels
    if df.shape[0] > n_rois:
        df = df.iloc[1:, :]

    matrix = df.values

    if matrix.shape != (n_rois, n_rois):
        print(f"⚠️ Unexpected shape {matrix.shape} in {os.path.basename(path)}")
    return matrix

# === Strength ===
def compute_strength(matrix):
    return np.sum(matrix, axis=1)

# === Clustering Coefficient ===
def compute_clustering(matrix):
    clustering = np.zeros(N_ROIS)
    matrix = matrix.copy()
    np.fill_diagonal(matrix, 0)
    matrix = (matrix + matrix.T) / 2
    for i in range(N_ROIS):
        neighbors = np.where(matrix[i] > 0)[0]
        k = len(neighbors)
        if k < 2:
            clustering[i] = 0.0
            continue
        subgraph = matrix[np.ix_(neighbors, neighbors)]
        numerator = np.sum((subgraph ** (1/3)) ** 3)
        clustering[i] = numerator / (k * (k - 1))
    return clustering

# === Compute and Save Metrics ===
def compute_and_save_metrics():
    for sid in SUBJECT_IDS:
        for cond in CONDITIONS:
            path = os.path.join(INPUT_DIR, f"{sid}_{cond}_cca.csv")
            if not os.path.exists(path):
                print(f"❌ Missing: {path}")
                continue
            matrix = load_cca_matrix_from_csv(path)
            strength = compute_strength(matrix)
            clustering = compute_clustering(matrix)
            pd.DataFrame([strength]).to_csv(f"{METRIC_OUTPUT_DIR}/{sid}_{cond}_strength.csv", index=False)
            pd.DataFrame([clustering]).to_csv(f"{METRIC_OUTPUT_DIR}/{sid}_{cond}_clustering.csv", index=False)
            print(f"✅ Metrics saved for {sid} ({cond})")

# === Plot Subject-wise Mean Comparison ===
def plot_metric_comparison():
    for metric in ["strength", "clustering"]:
        plt.figure(figsize=(10, 6))
        for cond in CONDITIONS:
            values = []
            for sid in SUBJECT_IDS:
                path = os.path.join(METRIC_OUTPUT_DIR, f"{sid}_{cond}_{metric}.csv")
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    values.append(df.values.mean())
                else:
                    values.append(np.nan)
            plt.plot(SUBJECT_IDS, values, marker='o', label=cond.capitalize())
        plt.title(f"Mean {metric.capitalize()} across Subjects (Task vs Rest)")
        plt.ylabel(metric.capitalize())
        plt.xlabel("Subject ID")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{FIGURE_OUTPUT_DIR}/mean_{metric}_comparison.png", dpi=300)
        plt.close()

# === Plot CCA Difference Heatmap ===
def plot_group_cca_difference():
    task_matrices, rest_matrices = [], []
    for sid in SUBJECT_IDS:
        task_path = os.path.join(INPUT_DIR, f"{sid}_task_cca.csv")
        rest_path = os.path.join(INPUT_DIR, f"{sid}_rest_cca.csv")
        if os.path.exists(task_path) and os.path.exists(rest_path):
            task_matrices.append(load_cca_matrix_from_csv(task_path))
            rest_matrices.append(load_cca_matrix_from_csv(rest_path))
    if task_matrices and rest_matrices:
        avg_task = np.mean(task_matrices, axis=0)
        avg_rest = np.mean(rest_matrices, axis=0)
        diff = avg_task - avg_rest
        plt.figure(figsize=(10, 8))
        sns.heatmap(diff, cmap="coolwarm", center=0)
        plt.title("CCA Difference (Task - Rest) Averaged Across Subjects")
        plt.xlabel("ROI j")
        plt.ylabel("ROI i")
        plt.tight_layout()
        plt.savefig(f"{FIGURE_OUTPUT_DIR}/group_cca_diff_heatmap.png", dpi=300)
        plt.close()
        print("✅ Group difference heatmap saved.")
    else:
        print("⚠️ Not enough data to compute group difference heatmap.")

# === Run All ===
if __name__ == "__main__":
    compute_and_save_metrics()
    plot_metric_comparison()
    plot_group_cca_difference()



# %%
