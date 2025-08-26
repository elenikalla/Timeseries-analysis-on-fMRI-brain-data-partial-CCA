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
# SUBJECT_IDS = [
# "sub-0001"
# ]
CONDITIONS = ["task", "rest"]
N_ROIS = 100
INPUT_DIR = "preprocessed data/correlation_csvs"
METRIC_OUTPUT_DIR = "preprocessed data/pcca_results/metrics"
FIGURE_OUTPUT_DIR = "preprocessed data/pcca_results/figures"

os.makedirs(METRIC_OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_OUTPUT_DIR, exist_ok=True)

# === Load PCCA tensor from CSV ===
def load_pcca_tensor_from_csv(path, n_rois=100):
    df = pd.read_csv(path)
    n_trials = df['trial'].max() + 1
    tensor = np.zeros((n_trials, n_rois, n_rois))
    for _, row in df.iterrows():
        t, i, j = int(row['trial']), int(row['roi_i']), int(row['roi_j'])
        tensor[t, i, j] = row['pcca_value']
    return tensor

# === Strength Metric ===
def compute_strength_per_trial(tensor):
    return np.sum(tensor, axis=2)

# === Clustering Metric ===
def compute_clustering_per_trial(tensor):
    n_trials, n_rois, _ = tensor.shape
    clustering_all = np.zeros((n_trials, n_rois))
    for t in range(n_trials):
        W = tensor[t].copy()
        np.fill_diagonal(W, 0)
        W = (W + W.T) / 2
        for i in range(n_rois):
            neighbors = np.where(W[i] > 0)[0]
            k = len(neighbors)
            if k < 2:
                clustering_all[t, i] = 0.0
                continue
            subgraph = W[np.ix_(neighbors, neighbors)]
            numerator = np.sum((subgraph ** (1/3)) ** 3)
            clustering_all[t, i] = numerator / (k * (k - 1))
    return clustering_all

# === Compute and save metrics ===
def compute_and_save_metrics():
    for sid in SUBJECT_IDS:
        for cond in CONDITIONS:
            csv_path = os.path.join(INPUT_DIR, f"{sid}_{cond}_pcca_3d.csv")
            tensor = load_pcca_tensor_from_csv(csv_path, N_ROIS)
            print(tensor.shape)
            strength = compute_strength_per_trial(tensor)
            clustering = compute_clustering_per_trial(tensor)
            pd.DataFrame(strength).to_csv(f"{METRIC_OUTPUT_DIR}/{sid}_{cond}_strength.csv", index=False)
            pd.DataFrame(clustering).to_csv(f"{METRIC_OUTPUT_DIR}/{sid}_{cond}_clustering.csv", index=False)
            print(f"✅ Metrics saved for {sid} ({cond})")

# === Plot subject-wise mean metric comparison ===
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

# === Load mean PCCA matrix ===
def load_mean_pcca_matrix(subject_id, condition):
    path = os.path.join(INPUT_DIR, f"{subject_id}_{condition}_pcca_3d.csv")
    df = pd.read_csv(path)
    matrix = np.zeros((N_ROIS, N_ROIS))
    counts = np.zeros((N_ROIS, N_ROIS))
    for _, row in df.iterrows():
        i, j = int(row['roi_i']), int(row['roi_j'])
        val = row['pcca_value']
        if np.isfinite(val):
            matrix[i, j] += val
            counts[i, j] += 1
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_matrix = np.divide(matrix, counts)
        mean_matrix[counts == 0] = 0
    return mean_matrix

def plot_group_pcca_difference():
    task_matrices, rest_matrices = [], []
    for sid in SUBJECT_IDS:
        task = load_mean_pcca_matrix(sid, "task")
        rest = load_mean_pcca_matrix(sid, "rest")
        print(f"{sid}: task={task.shape if task is not None else None}, rest={rest.shape if rest is not None else None}")
        if task is not None and rest is not None:
            task_matrices.append(task)
            rest_matrices.append(rest)

    print("N_task:", len(task_matrices), "N_rest:", len(rest_matrices))

    if task_matrices and rest_matrices:
        avg_task = np.mean(task_matrices, axis=0)
        avg_rest = np.mean(rest_matrices, axis=0)
        diff = avg_task - avg_rest
        plt.figure(figsize=(10, 8))
        sns.heatmap(diff, cmap="coolwarm", center=0)
        plt.title("PCCA Difference (Task - Rest) Averaged Across Subjects")
        plt.xlabel("ROI j")
        plt.ylabel("ROI i")
        plt.tight_layout()
        os.makedirs(FIGURE_OUTPUT_DIR, exist_ok=True)
        plt.savefig(f"{FIGURE_OUTPUT_DIR}/group_pcca_diff_heatmap.png", dpi=300)
        plt.close()
        print("Group difference heatmap saved.")
    else:
        print("Not enough data to compute group difference heatmap.")

# === Run all ===
if __name__ == "__main__":
    compute_and_save_metrics()
    plot_metric_comparison()
    plot_group_pcca_difference()

# %%
import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import fdrcorrection
from nilearn import plotting, datasets
from nilearn.maskers import NiftiLabelsMasker

# === Settings ===
SUBJECT_IDS = [
    "sub-0001", "sub-0002", "sub-0003", "sub-0004", "sub-0005",
    "sub-0006", "sub-0007", "sub-0008", "sub-0009", "sub-0011"
]
N_ROIS = 100
INPUT_DIR = "preprocessed data/correlation_csvs"

# === Load mean matrix ===
def load_mean_matrix(subject_id, condition):
    path = os.path.join(INPUT_DIR, f"{subject_id}_{condition}_pcca_3d.csv")
    df = pd.read_csv(path)
    matrix = np.zeros((N_ROIS, N_ROIS))
    counts = np.zeros((N_ROIS, N_ROIS))
    for _, row in df.iterrows():
        i, j = int(row['roi_i']), int(row['roi_j'])
        val = row['pcca_value']
        if np.isfinite(val):
            matrix[i, j] += val
            counts[i, j] += 1
    mean_matrix = np.divide(matrix, counts, where=counts != 0)
    return mean_matrix

# === Collect data ===
task_matrices, rest_matrices = [], []
for sid in SUBJECT_IDS:
    task_matrices.append(load_mean_matrix(sid, "task"))
    rest_matrices.append(load_mean_matrix(sid, "rest"))

# === Convert to arrays ===
task_stack = np.stack(task_matrices)
rest_stack = np.stack(rest_matrices)

# === Paired t-test ===
t_vals = np.zeros((N_ROIS, N_ROIS))
p_vals = np.ones((N_ROIS, N_ROIS))
for i in range(N_ROIS):
    for j in range(N_ROIS):
        t, p = ttest_rel(task_stack[:, i, j], rest_stack[:, i, j])
        t_vals[i, j] = t
        p_vals[i, j] = p

# === FDR correction ===
mask = np.triu(np.ones_like(p_vals), k=1).astype(bool)
_, pvals_corrected = fdrcorrection(p_vals[mask], alpha=0.05)
sig_matrix = np.zeros_like(p_vals)
sig_matrix[mask] = pvals_corrected < 0.05

# === Plot significant connectome ===
atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100, resolution_mm=1)
coords = plotting.find_parcellation_cut_coords(labels_img=atlas['maps'])

# Apply significance mask to mean difference
diff = task_stack.mean(axis=0) - rest_stack.mean(axis=0)
sig_diff = diff * sig_matrix
np.fill_diagonal(sig_diff, 0)

# Plot only significant edges
plotting.plot_connectome(
    sig_diff,
    coords,
    title="Significant PCCA Differences (Task - Rest)",
    edge_threshold="90%",
    node_size=20
)
plotting.show()



# %%
# === Histogram of t-values (upper triangle only) ===
import matplotlib.pyplot as plt

mask = np.triu_indices(N_ROIS, k=1)
upper_tvals = t_vals[mask]

# === Histogram of t-values (upper triangle only) ===
mask = np.triu_indices(N_ROIS, k=1)
upper_tvals = t_vals[mask]

plt.figure(figsize=(8, 5))
plt.hist(upper_tvals, bins=50, color='steelblue', edgecolor='black')
plt.title("Histogram of t-values across ROI pairs (PCCA Task vs Rest)")
plt.xlabel("t-value")
plt.ylabel("Frequency")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

# Ensure folder exists before saving
os.makedirs("preprocessed data/pcca_results/figures", exist_ok=True)
plt.savefig("preprocessed data/pcca_results/figures/tval_histogram_pcca_task_vs_rest.jpeg", dpi=300)
plt.show()

# %%
