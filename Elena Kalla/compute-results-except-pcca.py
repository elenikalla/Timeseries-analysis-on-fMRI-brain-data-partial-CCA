import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import fdrcorrection
from nilearn import datasets, plotting

# === Settings ===
SUBJECT_IDS = [
    "sub-0001", "sub-0002", "sub-0003", "sub-0004", "sub-0005",
    "sub-0006", "sub-0007", "sub-0008", "sub-0009", "sub-0011"
]
CONDITIONS = ["task", "rest"]
N_ROIS = 100

# === Network mapping (Schaefer-100 example) ===
NETWORK_MAP = {
    "Visual": list(range(0, 17)),
    "SomMot": list(range(17, 34)),
    "DorsAttn": list(range(34, 50)),
    "Salience": list(range(50, 60)),
    "Limbic": list(range(60, 70)),
    "Default": list(range(70, 90)),
    "Control": list(range(90, 100))
}

def roi_to_network(roi_index):
    for net_name, roi_list in NETWORK_MAP.items():
        if roi_index in roi_list:
            return net_name
    return "Unknown"

# === Update here for each type of correlation ===
# CORR_TYPES = ["cca", "pearson", "partial","pcca"]
CORR_TYPES = ["cca","pcca"]

BASE_INPUT_DIR = "preprocessed data/correlation_csvs"
BASE_OUTPUT_DIR = "preprocessed data/correlation_results"

# === Load matrix ===
def load_correlation_matrix(path, n_rois=100):
    df = pd.read_csv(path, header=None)
    if df.shape[1] > n_rois:
        df = df.iloc[:, 1:]
    if df.shape[0] > n_rois:
        df = df.iloc[1:, :]
    matrix = df.values
    if matrix.shape != (n_rois, n_rois):
        print(f"⚠️ Unexpected shape {matrix.shape} in {os.path.basename(path)}")
    return matrix


def plot_group_difference_heatmap(corr_type):
    input_dir = os.path.join(BASE_INPUT_DIR)
    figure_dir = os.path.join(BASE_OUTPUT_DIR, corr_type, "figures")
    os.makedirs(figure_dir, exist_ok=True)

    task_matrices, rest_matrices = [], []
    for sid in SUBJECT_IDS:
        task_path = os.path.join(input_dir, f"{sid}_task_{corr_type}.csv")
        rest_path = os.path.join(input_dir, f"{sid}_rest_{corr_type}.csv")
        if os.path.exists(task_path) and os.path.exists(rest_path):
            task = load_correlation_matrix(task_path)
            rest = load_correlation_matrix(rest_path)

            if task.shape == (N_ROIS, N_ROIS) and rest.shape == (N_ROIS, N_ROIS):
                task = np.nan_to_num(task, nan=0.0)
                rest = np.nan_to_num(rest, nan=0.0)
                task_matrices.append(task)
                rest_matrices.append(rest)
            else:
                print(f"⚠️ Skipping subject {sid} due to invalid shape.")

    if task_matrices and rest_matrices:
        task_stack = np.stack(task_matrices)
        rest_stack = np.stack(rest_matrices)

        t_vals = np.zeros((N_ROIS, N_ROIS))
        p_vals = np.ones((N_ROIS, N_ROIS))
        records = []
        significant_edges = []

        for i in range(N_ROIS):
            for j in range(N_ROIS):
                task_vals = task_stack[:, i, j]
                rest_vals = rest_stack[:, i, j]
                t, p = ttest_rel(task_vals, rest_vals, nan_policy='omit')
                t_vals[i, j] = t if np.isfinite(t) else 0.0
                p_vals[i, j] = p if np.isfinite(p) else 1.0

                records.append({
                    "ROI_i": i,
                    "ROI_j": j,
                    "t_value": t,
                    "p_value": p,
                    "task_vals": task_vals.tolist(),
                    "rest_vals": rest_vals.tolist()
                })

        df_debug = pd.DataFrame(records)
        csv_debug_path = os.path.join(figure_dir, "ttest_input_debug.csv")
        df_debug.to_csv(csv_debug_path, index=False)
        print(f"🧾 T-test inputs saved to: {csv_debug_path}")

        # FDR correction
        mask = np.triu(np.ones_like(p_vals), k=1).astype(bool)
        _, pvals_corrected = fdrcorrection(p_vals[mask], alpha=0.1)
        significant_mask = np.zeros_like(p_vals, dtype=bool)
        significant_mask[mask] = pvals_corrected < 0.1
        significant_mask = significant_mask | significant_mask.T

        avg_task = np.nan_to_num(task_stack.mean(axis=0), nan=0.0)
        avg_rest = np.nan_to_num(rest_stack.mean(axis=0), nan=0.0)
        diff = avg_task - avg_rest
        sig_diff = np.zeros_like(diff)
        sig_diff[significant_mask] = diff[significant_mask]
        np.fill_diagonal(sig_diff, 0)

        # Heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(sig_diff, cmap="coolwarm", center=0,
                    cbar_kws={'label': 'Δ Task-Rest'})
        plt.title(f"{corr_type.replace('_', ' ').title()} FDR-corrected Difference (Task - Rest)")
        plt.xlabel("ROI j")
        plt.ylabel("ROI i")
        plt.tight_layout()
        plt.savefig(os.path.join(figure_dir, f"group_{corr_type}_fdr_diff_heatmap.png"), dpi=300)
        plt.close()

        # Connectome plot
        atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100, resolution_mm=1)
        coords = plotting.find_parcellation_cut_coords(labels_img=atlas['maps'])
        plotting.plot_connectome(
            sig_diff,
            coords,
            title=f"{corr_type.replace('_', ' ').title()} Connectome (Task - Rest)",
            edge_threshold=0.2,
            node_size=20
        ).savefig(os.path.join(figure_dir, f"group_{corr_type}_connectome.png"), dpi=300)

        # Save significant edges with network labels
        for i in range(N_ROIS):
            for j in range(i + 1, N_ROIS):
                if significant_mask[i, j]:
                    significant_edges.append({
                        "ROI_i": i,
                        "ROI_j": j,
                        "Network_i": roi_to_network(i),
                        "Network_j": roi_to_network(j),
                        "Δ Task-Rest": diff[i, j],
                        "p-value": p_vals[i, j]
                    })

        df_edges = pd.DataFrame(significant_edges)
        df_edges.to_csv(os.path.join(figure_dir, f"top_fdr_edges_{corr_type}.csv"), index=False)
        print(f"✅ Group heatmap, connectome, and edge list saved for {corr_type}")
    else:
        print(f"⚠️ Not enough valid data for heatmap ({corr_type})")
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Metrics helpers (per matrix) ===
# def compute_strength(W):
#     Wsym = 0.5 * (W + W.T)
#     np.fill_diagonal(Wsym, 0.0)
#     return Wsym.sum(axis=1)  # (N_ROIS,)

def compute_strength(W):
    M = np.abs(W.copy())       # όπως έκανες
    # αν θες να ΜΗ μετράς διαγώνιο:
    # np.fill_diagonal(M, 0.0)
    return M.sum(axis=1)

# def compute_weighted_clustering_onnela(W):
#     Wsym = 0.5 * (W + W.T)
#     np.fill_diagonal(Wsym, 0.0)
#     n = Wsym.shape[0]
#     C = np.zeros(n, dtype=float)
#     for i in range(n):
#         nbrs = np.where(Wsym[i] > 0)[0]
#         k = len(nbrs)
#         if k < 2:
#             C[i] = 0.0
#             continue
#         sub = Wsym[np.ix_(nbrs, nbrs)]
#         A = np.power(np.maximum(sub, 0.0), 1/3)
#         triangles = np.trace(A @ A @ A)
#         C[i] = triangles / (k * (k - 1))
#     return C  # (N_ROIS,)
def compute_weighted_clustering_onnela(W):
    M = 0.5*(W + W.T)
    np.fill_diagonal(M, 0.0)
    N = M.shape[0]
    C = np.zeros(N)
    for i in range(N):
        nbrs = np.where(M[i] > 0)[0]
        k = len(nbrs)
        if k < 2:
            C[i] = 0.0
            continue
        sub = M[np.ix_(nbrs, nbrs)]
        # όπως είχες: (sub ** (1/3)) ** 3 == sub (για μη αρνητικά)
        numerator = np.sum(sub)
        C[i] = numerator / (k*(k-1))
    return C
# === 1) Υπολογισμός & αποθήκευση metrics ανά subject/condition/corr_type ===
def compute_and_save_metrics(corr_type):
    metrics_dir = os.path.join(BASE_OUTPUT_DIR, corr_type, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    print(f"📊 Computing strength & clustering for corr_type={corr_type} ...")
    for sid in SUBJECT_IDS:
        for cond in CONDITIONS:
            in_path = os.path.join(BASE_INPUT_DIR, f"{sid}_{cond}_{corr_type}.csv")
            if not os.path.exists(in_path):
                print(f"⚠️ Missing file: {in_path}")
                continue
            W = load_correlation_matrix(in_path, n_rois=N_ROIS)
            W = np.nan_to_num(W, nan=0.0)

            strength = compute_strength(W)                      # (N_ROIS,)
            clustering = compute_weighted_clustering_onnela(W)  # (N_ROIS,)

            pd.DataFrame({"strength": strength}).to_csv(
                os.path.join(metrics_dir, f"{sid}_{cond}_strength.csv"),
                index=False
            )
            pd.DataFrame({"clustering": clustering}).to_csv(
                os.path.join(metrics_dir, f"{sid}_{cond}_clustering.csv"),
                index=False
            )
            print(f"✅ Saved {sid} ({cond}) metrics → {metrics_dir}")
    print("✅ Done.\n")

# === 2) Plot: mean metric per subject (Task vs Rest) για συγκεκριμένο corr_type ===
def plot_metric_comparison(corr_type):
    metrics_dir = os.path.join(BASE_OUTPUT_DIR, corr_type, "metrics")
    figs_dir = os.path.join(BASE_OUTPUT_DIR, corr_type, "figures")
    os.makedirs(figs_dir, exist_ok=True)

    print(f"📈 Plotting mean metrics for corr_type={corr_type} ...")
    for metric in ["strength", "clustering"]:
        plt.figure(figsize=(10, 6))
        for cond in CONDITIONS:
            vals = []
            for sid in SUBJECT_IDS:
                path = os.path.join(metrics_dir, f"{sid}_{cond}_{metric}.csv")
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    vals.append(df.iloc[:, 0].mean())
                else:
                    vals.append(np.nan)
            plt.plot(SUBJECT_IDS, vals, marker="o", label=cond.capitalize())
        plt.title(f"{corr_type.upper()} — Mean {metric.capitalize()} across Subjects (Task vs Rest)")
        plt.ylabel(metric.capitalize())
        plt.xlabel("Subject ID")
        plt.xticks(rotation=45)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        out = os.path.join(figs_dir, f"mean_{metric}_comparison.png")
        plt.savefig(out, dpi=300)
        plt.close()
        print(f"✅ Saved: {out}")
    print("✅ Done.\n")

# === helper για per-network σύνοψη ===
def summarize_per_network(metric_matrix_subjects_by_roi):
    """
    metric_matrix_subjects_by_roi: (S, N_ROIS)
    επιστρέφει dict {network: vector(S,) με τον μέσο όρο ανά subject στα ROIs του network}
    """
    nets = {}
    for net, rois in NETWORK_MAP.items():
        nets[net] = metric_matrix_subjects_by_roi[:, rois].mean(axis=1)
    return nets

# === 3) Plot: network-wise barplots για strength/clustering (corr_type-specific) ===
def plot_network_comparisons(corr_type):
    metrics_dir = os.path.join(BASE_OUTPUT_DIR, corr_type, "metrics")
    figs_dir = os.path.join(BASE_OUTPUT_DIR, corr_type, "figures")
    os.makedirs(figs_dir, exist_ok=True)

    print(f"📊 Building network-wise barplots for corr_type={corr_type} ...")
    for metric in ["strength", "clustering"]:
        task_rows, rest_rows = [], []
        colname = "strength" if metric == "strength" else "clustering"

        for sid in SUBJECT_IDS:
            tpath = os.path.join(metrics_dir, f"{sid}_task_{metric}.csv")
            rpath = os.path.join(metrics_dir, f"{sid}_rest_{metric}.csv")
            if os.path.exists(tpath) and os.path.exists(rpath):
                task_rows.append(pd.read_csv(tpath)[colname].values)
                rest_rows.append(pd.read_csv(rpath)[colname].values)

        if not task_rows or not rest_rows:
            print(f"⚠️ Not enough data to plot network comparisons for {metric}.")
            continue

        task_mat = np.stack(task_rows)  # (S, N_ROIS)
        rest_mat = np.stack(rest_rows)  # (S, N_ROIS)
        task_nets = summarize_per_network(task_mat)
        rest_nets = summarize_per_network(rest_mat)

        plt.figure(figsize=(12, 6))
        idx = np.arange(len(NETWORK_MAP))
        bar_w = 0.35
        task_vals = [np.mean(task_nets[net]) for net in NETWORK_MAP]
        rest_vals = [np.mean(rest_nets[net]) for net in NETWORK_MAP]

        plt.bar(idx, task_vals, bar_w, label="Task")
        plt.bar(idx + bar_w, rest_vals, bar_w, label="Rest")
        plt.xticks(idx + bar_w / 2, list(NETWORK_MAP.keys()), rotation=30)
        plt.ylabel(f"Mean {metric.capitalize()}")
        plt.title(f"{corr_type.upper()} — Network-wise {metric.capitalize()} Comparison")
        plt.legend()
        plt.tight_layout()
        out = os.path.join(figs_dir, f"networkwise_{metric}.png")
        plt.savefig(out, dpi=300)
        plt.close()
        print(f"✅ Saved: {out}")
    print("✅ Done.\n")

# === Main Runner ===
if __name__ == "__main__":
    for corr_type in CORR_TYPES:
        compute_and_save_metrics(corr_type)
        plot_metric_comparison(corr_type)
        plot_network_comparisons(corr_type)
        plot_group_difference_heatmap(corr_type)
