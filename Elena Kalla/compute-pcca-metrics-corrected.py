# %%
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
INPUT_DIR = "preprocessed data/correlation_csvs"          # περιμένει αρχεία: {sid}_{cond}_pcca.csv (long-format)
METRIC_OUTPUT_DIR = "preprocessed data/pcca_results/metrics"
FIGURE_OUTPUT_DIR = "preprocessed data/pcca_results/figures"

os.makedirs(METRIC_OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_OUTPUT_DIR, exist_ok=True)

# === Load PCCA matrix from CSV (condition-wise, long format: roi_i, roi_j, pcca_value) ===
def load_pcca_matrix_from_csv(path, n_rois=100):
    """
    Αναμένει CSV με στήλες: roi_i, roi_j, pcca_value (ΧΩΡΙΣ trial).
    Επιστρέφει W (n_rois, n_rois).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path)
    W = np.zeros((n_rois, n_rois), dtype=float)
    for _, row in df.iterrows():
        i, j = int(row['roi_i']), int(row['roi_j'])
        val = float(row['pcca_value'])
        if np.isfinite(val):
            W[i, j] = val
    # Συμμετρία αν χρειάζεται
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 1.0)
    return W

# === Metrics (per matrix) ===
def compute_strength(W):
    """
    Weighted degree per ROI (μηδενίζουμε διαγώνιο και συμμετροποιούμε).
    Επιστρέφει vector (n_rois,).
    """
    Wsym = 0.5 * (W + W.T)
    np.fill_diagonal(Wsym, 0.0)
    return Wsym.sum(axis=1)

def compute_weighted_clustering_onnela(W):
    """
    Weighted clustering coefficient (Onnela et al. 2005).
    Για κάθε κόμβο: A = W^(1/3) element-wise, C_i = (A^3)_ii / (k_i*(k_i-1))
    Επιστρέφει vector (n_rois,).
    """
    Wsym = 0.5 * (W + W.T)
    np.fill_diagonal(Wsym, 0.0)
    n = Wsym.shape[0]
    C = np.zeros(n, dtype=float)

    # γείτονες ανά κόμβο
    for i in range(n):
        neighbors = np.where(Wsym[i] > 0)[0]
        k = len(neighbors)
        if k < 2:
            C[i] = 0.0
            continue
        sub = Wsym[np.ix_(neighbors, neighbors)]
        A = np.power(np.maximum(sub, 0.0), 1/3)  # element-wise 1/3
        triangles = np.trace(A @ A @ A)
        C[i] = triangles / (k * (k - 1))
    return C

# === Compute and save metrics per subject/condition (condition-wise) ===
def compute_and_save_metrics():
    for sid in SUBJECT_IDS:
        for cond in CONDITIONS:
            csv_path = os.path.join(INPUT_DIR, f"{sid}_{cond}_pcca.csv")  # ΝΕΟ όνομα αρχείου (χωρίς _3d)
            try:
                W = load_pcca_matrix_from_csv(csv_path, N_ROIS)
            except FileNotFoundError as e:
                print(e)
                continue

            strength = compute_strength(W)                           # (N_ROIS,)
            clustering = compute_weighted_clustering_onnela(W)       # (N_ROIS,)

            pd.DataFrame(strength, columns=["strength"]).to_csv(
                f"{METRIC_OUTPUT_DIR}/{sid}_{cond}_strength.csv", index=False
            )
            pd.DataFrame(clustering, columns=["clustering"]).to_csv(
                f"{METRIC_OUTPUT_DIR}/{sid}_{cond}_clustering.csv", index=False
            )
            print(f"✅ Metrics saved for {sid} ({cond})")

# === Plot subject-wise mean metric comparison (Task vs Rest) ===
def plot_metric_comparison():
    for metric in ["strength", "clustering"]:
        plt.figure(figsize=(10, 6))
        for cond in CONDITIONS:
            values = []
            for sid in SUBJECT_IDS:
                path = os.path.join(METRIC_OUTPUT_DIR, f"{sid}_{cond}_{metric}.csv")
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    # μέσος όρος πάνω στα ROIs
                    values.append(df.iloc[:, 0].mean())
                else:
                    values.append(np.nan)
            plt.plot(SUBJECT_IDS, values, marker='o', label=cond.capitalize())
        plt.title(f"Mean {metric.capitalize()} across Subjects (Task vs Rest)")
        plt.ylabel(metric.capitalize())
        plt.xlabel("Subject ID")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        out = f"{FIGURE_OUTPUT_DIR}/mean_{metric}_comparison.png"
        plt.savefig(out, dpi=300)
        plt.close()
        print(f"📈 Saved: {out}")

# === Load condition-wise PCCA matrix (helper για group plots) ===
def load_pcca_matrix(subject_id, condition):
    path = os.path.join(INPUT_DIR, f"{subject_id}_{condition}_pcca.csv")
    if not os.path.exists(path):
        return None
    return load_pcca_matrix_from_csv(path, N_ROIS)

# === Group heatmap: Task - Rest ===
def plot_group_pcca_difference():
    task_matrices, rest_matrices = [], []
    for sid in SUBJECT_IDS:
        task = load_pcca_matrix(sid, "task")
        rest = load_pcca_matrix(sid, "rest")
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
        out = f"{FIGURE_OUTPUT_DIR}/group_pcca_diff_heatmap.png"
        plt.savefig(out, dpi=300)
        plt.close()
        print(f"🗺️ Group difference heatmap saved: {out}")
    else:
        print("Not enough data to compute group difference heatmap.")

# === Run all ===
if __name__ == "__main__":
    compute_and_save_metrics()
    plot_metric_comparison()
    plot_group_pcca_difference()

# %%  ------------------  STATS & CONNECTOME PLOT  ------------------
import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import fdrcorrection
from nilearn import plotting, datasets

# === Settings (repeat for standalone run of this cell) ===
SUBJECT_IDS = [
    "sub-0001", "sub-0002", "sub-0003", "sub-0004", "sub-0005",
    "sub-0006", "sub-0007", "sub-0008", "sub-0009", "sub-0011"
]
N_ROIS = 100
INPUT_DIR = "preprocessed data/correlation_csvs"

# === Load matrices stacks ===
def load_stack(condition):
    mats = []
    for sid in SUBJECT_IDS:
        W = load_pcca_matrix(sid, condition)
        if W is not None:
            mats.append(W)
    if not mats:
        raise RuntimeError(f"No matrices found for condition: {condition}")
    return np.stack(mats)   # (S, N, N)

task_stack = load_stack("task")
rest_stack = load_stack("rest")

# === Paired t-test per edge across subjects ===
S = task_stack.shape[0]
t_vals = np.zeros((N_ROIS, N_ROIS))
p_vals = np.ones((N_ROIS, N_ROIS))

for i in range(N_ROIS):
    for j in range(N_ROIS):
        t, p = ttest_rel(task_stack[:, i, j], rest_stack[:, i, j], nan_policy='omit')
        t_vals[i, j] = t
        p_vals[i, j] = p

# === FDR correction on upper triangle ===
tri_mask = np.triu(np.ones_like(p_vals, dtype=bool), k=1)
p_triu = p_vals[tri_mask]
rej, p_corr = fdrcorrection(p_triu, alpha=0.05)

sig_matrix = np.zeros_like(p_vals, dtype=bool)
sig_matrix[tri_mask] = rej
sig_matrix = np.logical_or(sig_matrix, sig_matrix.T)   # συμμετρία
np.fill_diagonal(sig_matrix, False)

# === Significant difference matrix ===
diff = np.nanmean(task_stack, axis=0) - np.nanmean(rest_stack, axis=0)
sig_diff = np.where(sig_matrix, diff, 0.0)
np.fill_diagonal(sig_diff, 0.0)

# === Plot significant connectome ===
atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100, resolution_mm=1)
coords = plotting.find_parcellation_cut_coords(labels_img=atlas['maps'])

plotting.plot_connectome(
    sig_diff,
    coords,
    title="Significant PCCA Differences (Task - Rest)",
    edge_threshold="90%",
    node_size=20
)
plotting.show()

# %%  ------------------  HISTOGRAM OF T-VALUES  ------------------
import matplotlib.pyplot as plt
tri_idx = np.triu_indices(N_ROIS, k=1)
upper_tvals = t_vals[tri_idx]

plt.figure(figsize=(8, 5))
plt.hist(upper_tvals, bins=50, color='steelblue', edgecolor='black')
plt.title("Histogram of t-values across ROI pairs (PCCA Task vs Rest)")
plt.xlabel("t-value")
plt.ylabel("Frequency")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
os.makedirs(FIGURE_OUTPUT_DIR, exist_ok=True)
plt.savefig(f"{FIGURE_OUTPUT_DIR}/tval_histogram_pcca_task_vs_rest.jpeg", dpi=300)
plt.show()
