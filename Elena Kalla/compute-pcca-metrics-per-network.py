import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import fdrcorrection

# === Settings ===
SUBJECT_IDS = [
    "sub-0001", "sub-0002", "sub-0003", "sub-0004", "sub-0005",
    "sub-0006", "sub-0007", "sub-0008", "sub-0009", "sub-0011"
]
CONDITIONS = ["task", "rest"]
N_ROIS = 100
INPUT_DIR = "preprocessed data/correlation_csvs"
METRIC_OUTPUT_DIR = "preprocessed data/pcca_results_per_nw/metrics"
FIGURE_OUTPUT_DIR = "preprocessed data/pcca_results_per_nw/figures"

os.makedirs(METRIC_OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_OUTPUT_DIR, exist_ok=True)

# === ROI to network mapping (Schaefer-100 example) ===
NETWORK_MAP = {
    "Visual": list(range(0, 17)),
    "SomMot": list(range(17, 34)),
    "DorsAttn": list(range(34, 50)),
    "Salience": list(range(50, 60)),
    "Limbic": list(range(60, 70)),
    "Default": list(range(70, 90)),
    "Control": list(range(90, 100))
}

def load_pcca_tensor_from_csv(path, n_rois=100):
    df = pd.read_csv(path)
    n_trials = df['trial'].max() + 1
    tensor = np.zeros((n_trials, n_rois, n_rois))
    for _, row in df.iterrows():
        t, i, j = int(row['trial']), int(row['roi_i']), int(row['roi_j'])
        tensor[t, i, j] = row['pcca_value']
    return tensor

def compute_strength_per_trial(tensor):
    return np.sum(np.abs(tensor), axis=2)

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

def compute_and_save_metrics():
    print("📊 Computing strength and clustering metrics per subject...")
    for sid in SUBJECT_IDS:
        for cond in CONDITIONS:
            csv_path = os.path.join(INPUT_DIR, f"{sid}_{cond}_pcca_3d.csv")
            if not os.path.exists(csv_path):
                print(f"⚠️ Missing file: {csv_path}")
                continue
            tensor = load_pcca_tensor_from_csv(csv_path, N_ROIS)
            strength = compute_strength_per_trial(tensor)
            clustering = compute_clustering_per_trial(tensor)
            pd.DataFrame(strength).to_csv(f"{METRIC_OUTPUT_DIR}/{sid}_{cond}_strength.csv", index=False)
            pd.DataFrame(clustering).to_csv(f"{METRIC_OUTPUT_DIR}/{sid}_{cond}_clustering.csv", index=False)
            print(f"✅ Saved metrics for {sid} ({cond})")
    print("✅ All metrics computed and saved.\n")

def plot_metric_comparison():
    print("📈 Plotting mean strength and clustering comparisons across subjects...")
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
        plt.xticks(rotation=45)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        output_path = f"{FIGURE_OUTPUT_DIR}/mean_{metric}_comparison.png"
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"✅ Saved: {output_path}")
    print("✅ All metric comparison plots completed.\n")

def summarize_per_network(metric_data):
    network_means = {}
    for net, rois in NETWORK_MAP.items():
        network_means[net] = metric_data[:, rois].mean(axis=1)
    return network_means

def plot_network_comparisons():
    print("📊 Creating network-wise strength/clustering comparison plots...")
    for metric in ["strength", "clustering"]:
        task_all, rest_all = [], []
        for sid in SUBJECT_IDS:
            task_path = f"{METRIC_OUTPUT_DIR}/{sid}_task_{metric}.csv"
            rest_path = f"{METRIC_OUTPUT_DIR}/{sid}_rest_{metric}.csv"
            if os.path.exists(task_path) and os.path.exists(rest_path):
                task = pd.read_csv(task_path).values.mean(axis=0)
                rest = pd.read_csv(rest_path).values.mean(axis=0)
                task_all.append(task)
                rest_all.append(rest)
        if not task_all or not rest_all:
            print(f"⚠️ Not enough data to plot {metric} network comparisons.")
            continue
        task_all = np.stack(task_all)
        rest_all = np.stack(rest_all)
        task_nets = summarize_per_network(task_all)
        rest_nets = summarize_per_network(rest_all)
        plt.figure(figsize=(12, 6))
        bar_width = 0.35
        index = np.arange(len(NETWORK_MAP))
        task_vals = [np.mean(task_nets[net]) for net in NETWORK_MAP]
        rest_vals = [np.mean(rest_nets[net]) for net in NETWORK_MAP]
        plt.bar(index, task_vals, bar_width, label="Task")
        plt.bar(index + bar_width, rest_vals, bar_width, label="Rest")
        plt.xticks(index + bar_width / 2, list(NETWORK_MAP.keys()), rotation=30)
        plt.ylabel(f"Mean {metric.capitalize()}")
        plt.title(f"Network-wise {metric.capitalize()} Comparison")
        plt.legend()
        plt.tight_layout()
        output_path = f"{FIGURE_OUTPUT_DIR}/networkwise_{metric}.png"
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"✅ Saved: {output_path}")
    print("✅ All network comparison plots completed.\n")

def plot_task_rest_heatmap():
    print("🧠 Creating task-rest average PCCA heatmap...")
    task_all, rest_all = [], []
    for sid in SUBJECT_IDS:
        task_path = f"{INPUT_DIR}/{sid}_task_pcca_3d.csv"
        rest_path = f"{INPUT_DIR}/{sid}_rest_pcca_3d.csv"
        if not os.path.exists(task_path) or not os.path.exists(rest_path):
            continue
        task_tensor = load_pcca_tensor_from_csv(task_path)
        rest_tensor = load_pcca_tensor_from_csv(rest_path)
        task_all.append(np.nanmean(task_tensor, axis=0))
        rest_all.append(np.nanmean(rest_tensor, axis=0))

    if not task_all or not rest_all:
        print("⚠️ Not enough data for heatmap.")
        return

    group_task = np.mean(task_all, axis=0)
    group_rest = np.mean(rest_all, axis=0)
    diff = group_task - group_rest

    plt.figure(figsize=(10, 8))
    sns.heatmap(diff, cmap="coolwarm", center=0, cbar_kws={'label': 'PCCA Difference (Task - Rest)'})
    plt.title("Mean PCCA Difference (Task - Rest) across Subjects")
    plt.xlabel("ROI j")
    plt.ylabel("ROI i")
    plt.tight_layout()
    output_path = f"{FIGURE_OUTPUT_DIR}/pcca_difference_heatmap.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Saved: {output_path}\n")

def perform_edgewise_ttest():
    print("🧪 Performing edge-wise t-tests with FDR correction...")
    from nilearn import datasets, plotting

    # === Load and stack task/rest PCCA mean matrices safely ===
    task_stack, rest_stack = [], []
    for sid in SUBJECT_IDS:
        task_path = f"{INPUT_DIR}/{sid}_task_pcca_3d.csv"
        rest_path = f"{INPUT_DIR}/{sid}_rest_pcca_3d.csv"
        if os.path.exists(task_path) and os.path.exists(rest_path):
            task_tensor = load_pcca_tensor_from_csv(task_path)
            rest_tensor = load_pcca_tensor_from_csv(rest_path)
            task_mean = np.nanmean(task_tensor, axis=0)
            rest_mean = np.nanmean(rest_tensor, axis=0)
            task_mean = np.nan_to_num(task_mean, nan=0.0)
            rest_mean = np.nan_to_num(rest_mean, nan=0.0)
            task_stack.append(task_mean)
            rest_stack.append(rest_mean)

    task_stack = np.stack(task_stack)
    rest_stack = np.stack(rest_stack)

    # === Paired t-test per edge + log inputs ===
    t_vals = np.zeros((N_ROIS, N_ROIS))
    p_vals = np.ones((N_ROIS, N_ROIS))
    records = []

    for i in range(N_ROIS):
        for j in range(N_ROIS):
            task_vals = task_stack[:, i, j]
            rest_vals = rest_stack[:, i, j]
            t, p = ttest_rel(task_vals, rest_vals)
            t_vals[i, j] = t
            p_vals[i, j] = p

            records.append({
                "ROI_i": i,
                "ROI_j": j,
                "t_value": t,
                "p_value": p,
                "task_vals": task_vals.tolist(),
                "rest_vals": rest_vals.tolist()
            })

    df_debug = pd.DataFrame(records)
    csv_debug_path = os.path.join(FIGURE_OUTPUT_DIR, "ttest_input_debug.csv")
    df_debug.to_csv(csv_debug_path, index=False)
    print(f"🧾 T-test inputs saved to: {csv_debug_path}")

    # === FDR correction only for upper triangle ===
    mask = np.triu(np.ones_like(p_vals), k=1).astype(bool)
    _, pvals_corrected = fdrcorrection(p_vals[mask], alpha=0.3)
    sig_matrix = np.zeros_like(p_vals, dtype=bool)
    sig_matrix[mask] = pvals_corrected < 0.3
    sig_matrix = sig_matrix | sig_matrix.T  # symmetric

    # === Compute difference and apply mask safely ===
    diff = np.nan_to_num(task_stack.mean(axis=0) - rest_stack.mean(axis=0), nan=0.0)
    sig_diff = diff * sig_matrix
    np.fill_diagonal(sig_diff, 0)

    # === Heatmap of significant differences
    heatmap_path = f"{FIGURE_OUTPUT_DIR}/pcca_connectome_fdr_corrected.png"
    plt.figure(figsize=(10, 8))
    sns.heatmap(sig_diff, cmap="coolwarm", center=0,
                cbar_kws={'label': 'Δ PCCA (Task - Rest)'})
    plt.title("Significant Edge-wise PCCA Differences (FDR-corrected)")
    plt.xlabel("ROI j")
    plt.ylabel("ROI i")
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print(f"✅ Saved heatmap: {heatmap_path}")

    # === Connectome plot
    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100, resolution_mm=1)
    coords = plotting.find_parcellation_cut_coords(labels_img=atlas['maps'])

    connectome_path = f"{FIGURE_OUTPUT_DIR}/pcca_connectome_graph.png"
    plotting.plot_connectome(
        sig_diff,
        coords,
        title="Significant PCCA Differences (Task - Rest)",
        node_size=20
    ).savefig(connectome_path, dpi=300)
    print(f"✅ Saved connectome graph: {connectome_path}")

    # === Save significant edges as CSV ===
    def roi_to_network(roi_index):
        for net_name, roi_list in NETWORK_MAP.items():
            if roi_index in roi_list:
                return net_name
        return "Unknown"

    edges = []
    for i in range(N_ROIS):
        for j in range(i + 1, N_ROIS):
            if sig_matrix[i, j]:
                net_i = roi_to_network(i)
                net_j = roi_to_network(j)
                edges.append({
                    "ROI_i": i,
                    "ROI_j": j,
                    "Network_i": net_i,
                    "Network_j": net_j,
                    "Δ Task-Rest": diff[i, j],
                    "p-value": p_vals[i, j]
                })

    df_edges = pd.DataFrame(edges)

    csv_path = f"{FIGURE_OUTPUT_DIR}/top_fdr_edges.csv"
    df_edges.to_csv(csv_path, index=False)
    print(f"✅ Saved edge list: {csv_path}\n")

if __name__ == "__main__":
    # compute_and_save_metrics()
    # plot_metric_comparison()
    # plot_network_comparisons()
    # plot_task_rest_heatmap()
    perform_edgewise_ttest()
