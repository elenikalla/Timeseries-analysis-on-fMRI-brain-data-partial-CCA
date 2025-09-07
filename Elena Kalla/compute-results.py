# -*- coding: utf-8 -*-
import os
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel, ttest_1samp, chi2
from statsmodels.stats.multitest import fdrcorrection
from nilearn import datasets, plotting

# =========================
# Settings
# =========================
SUBJECT_IDS = [
    "sub-0001","sub-0002","sub-0003","sub-0004","sub-0005",
    "sub-0006","sub-0007","sub-0008","sub-0009","sub-0011"
]
CONDITIONS = ["task", "rest"]
N_ROIS = 100
# Βάλε ό,τι θέλεις να τρέξει
CORR_TYPES = ["pearson","partial","cca","pcca"]

WILKS_PX = 3
WILKS_PY = 3
WILKS_N_EFF = 40 #38 gia pcca

ALPHA_FDR = 0.10      # q για FDR-corrected
ALPHA_UNCORR = 0.05   # p για uncorrected

# Paths
HERE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
BASE_INPUT_DIR  = str(HERE / "preprocessed data" / "correlation_csvs")
BASE_OUTPUT_DIR = str(HERE / "preprocessed data" / "correlation_results")

# =========================
# Parcellation / Networks
# =========================
NETWORK_MAP = {
    "Visual": list(range(0, 17)),
    "SomMot": list(range(17, 34)),
    "DorsAttn": list(range(34, 50)),
    "Salience": list(range(50, 60)),
    "Limbic": list(range(60, 70)),
    "Default": list(range(70, 90)),
    "Control": list(range(90, 100)),
}

def roi_to_network(roi_index: int) -> str:
    for net_name, roi_list in NETWORK_MAP.items():
        if roi_index in roi_list:
            return net_name
    return "Unknown"

# =========================
# Helpers
# =========================
def corr_fname(sid: str, cond: str, corr_type: str) -> str:
    return f"{sid}_{cond}_{corr_type}.csv"

def corr_path(input_dir: str, sid: str, cond: str, corr_type: str) -> str:
    return os.path.join(input_dir, corr_fname(sid, cond, corr_type))

def load_correlation_matrix(path: str, n_rois: int = N_ROIS) -> np.ndarray:
    df = pd.read_csv(path, header=None)
    # κόψε header/labels αν υπάρχουν
    if df.shape[1] > n_rois:
        df = df.iloc[:, 1:]
    if df.shape[0] > n_rois:
        df = df.iloc[1:, :]
    M = df.values
    if M.shape != (n_rois, n_rois):
        print(f"⚠️ Unexpected shape {M.shape} in {os.path.basename(path)}")
    return M

def fisher_z(r):
    r = np.clip(r, -0.999999, 0.999999)
    return np.arctanh(r)

def atanh_cca(g):
    # κανονικές συσχέτισεις ∈ [0,1)
    g = np.clip(g, 1e-12, 0.999999)
    return np.arctanh(g)

def z_transform_for_test(x: np.ndarray, corr_type: str) -> np.ndarray:
    """
    Μετασχηματισμός πριν από t-test:
    - Pearson/Partial: Fisher z
    - CCA/PCCA: atanh(γ) (μονοτονικός, σταθεροποιεί διακύμανση γύρω από 0+)
    """
    if corr_type in ("pearson","partial"):
        return fisher_z(x)
    elif corr_type in ("cca","pcca"):
        return atanh_cca(x)
    else:
        return x

def load_stacks(corr_type: str):
    """
    Επιστρέφει (task_stack, rest_stack, used_subjects) με σχήμα (S, N_ROIS, N_ROIS).
    """
    task_list, rest_list, used_sids = [], [], []
    for sid in SUBJECT_IDS:
        tp = corr_path(BASE_INPUT_DIR, sid, "task", corr_type)
        rp = corr_path(BASE_INPUT_DIR, sid, "rest", corr_type)
        if not (os.path.exists(tp) and os.path.exists(rp)):
            continue
        t = load_correlation_matrix(tp, n_rois=N_ROIS)
        r = load_correlation_matrix(rp, n_rois=N_ROIS)
        if t.shape != (N_ROIS, N_ROIS) or r.shape != (N_ROIS, N_ROIS):
            continue
        task_list.append(np.nan_to_num(t, nan=0.0))
        rest_list.append(np.nan_to_num(r, nan=0.0))
        used_sids.append(sid)
    if len(task_list) == 0:
        return None, None, []
    return np.stack(task_list), np.stack(rest_list), used_sids

# =========================
# Graph metrics (όπως πριν)
# =========================
def compute_strength(W: np.ndarray) -> np.ndarray:
    M = np.abs(W.copy())
    return M.sum(axis=1)

def compute_weighted_clustering_onnela(W: np.ndarray) -> np.ndarray:
    M = 0.5 * (W + W.T)
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
        numerator = np.sum(sub)  # placeholder (όχι ακριβής Onnela)
        C[i] = numerator / (k * (k - 1))
    return C

# =========================
# (Προαιρετικό) Save metrics & plots
# =========================
def compute_and_save_metrics(corr_type: str) -> None:
    metrics_dir = os.path.join(BASE_OUTPUT_DIR, corr_type, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    print(f"📊 Computing strength & clustering for corr_type={corr_type} ...")
    for sid in SUBJECT_IDS:
        for cond in CONDITIONS:
            in_path = corr_path(BASE_INPUT_DIR, sid, cond, corr_type)
            if not os.path.exists(in_path): continue
            W = load_correlation_matrix(in_path, n_rois=N_ROIS)
            W = np.nan_to_num(W, nan=0.0)

            strength = compute_strength(W)
            clustering = compute_weighted_clustering_onnela(W)

            pd.DataFrame({"strength": strength}).to_csv(
                os.path.join(metrics_dir, f"{sid}_{cond}_strength.csv"), index=False)
            pd.DataFrame({"clustering": clustering}).to_csv(
                os.path.join(metrics_dir, f"{sid}_{cond}_clustering.csv"), index=False)
    print("✅ Done.\n")

def plot_metric_comparison(corr_type: str) -> None:
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
        plt.ylabel(metric.capitalize()); plt.xlabel("Subject ID")
        plt.xticks(rotation=45); plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(); plt.tight_layout()
        out = os.path.join(figs_dir, f"mean_{metric}_comparison.png")
        plt.savefig(out, dpi=300); plt.close()
        print(f"✅ Saved: {out}")
    print("✅ Done.\n")

def summarize_per_network(metric_matrix_subjects_by_roi: np.ndarray) -> dict:
    nets = {}
    for net, rois in NETWORK_MAP.items():
        nets[net] = metric_matrix_subjects_by_roi[:, rois].mean(axis=1)
    return nets

def plot_network_comparisons(corr_type: str) -> None:
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

        task_mat = np.stack(task_rows)
        rest_mat = np.stack(rest_rows)
        task_nets = summarize_per_network(task_mat)
        rest_nets = summarize_per_network(rest_mat)

        plt.figure(figsize=(12, 6))
        idx = np.arange(len(NETWORK_MAP)); bar_w = 0.35
        task_vals = [np.mean(task_nets[net]) for net in NETWORK_MAP]
        rest_vals = [np.mean(rest_nets[net]) for net in NETWORK_MAP]

        plt.bar(idx, task_vals, bar_w, label="Task")
        plt.bar(idx + bar_w, rest_vals, bar_w, label="Rest")
        plt.xticks(idx + bar_w / 2, list(NETWORK_MAP.keys()), rotation=30)
        plt.ylabel(f"Mean {metric.capitalize()}")
        plt.title(f"{corr_type.upper()} — Network-wise {metric.capitalize()} Comparison")
        plt.legend(); plt.tight_layout()
        out = os.path.join(figs_dir, f"networkwise_{metric}.png")
        plt.savefig(out, dpi=300); plt.close()
        print(f"✅ Saved: {out}")
    print("✅ Done.\n")

# --- helpers: inverse of the z-transform you ήδη χρησιμοποιείς ---
def z_transform_for_hist(x: np.ndarray, corr_type: str) -> np.ndarray:
    # ίδιο με z_transform_for_test, απλώς όνομα για σαφήνεια
    return z_transform_for_test(x, corr_type)

def inv_z(y_z: np.ndarray) -> np.ndarray:
    # και για Fisher-z και για atanh(γ) το inverse είναι tanh
    return np.tanh(y_z)

# =========================
# 4A) Κατανομές (raw & normalized) & 95% CI (στο z-space)
# =========================
def distributions_and_ci(corr_type: str) -> None:
    task, rest, used = load_stacks(corr_type)
    if task is None:
        print(f"⚠️ No data for {corr_type}"); return

    figs_dir  = os.path.join(BASE_OUTPUT_DIR, corr_type, "figures")
    stats_dir = os.path.join(BASE_OUTPUT_DIR, corr_type, "stats")
    os.makedirs(figs_dir, exist_ok=True); os.makedirs(stats_dir, exist_ok=True)

    _, N, _ = task.shape
    iu = np.triu_indices(N, 1)

    # -------- RAW values (όπως πριν) --------
    vals_task_raw  = task[:, iu[0], iu[1]].ravel()
    vals_rest_raw  = rest[:, iu[0], iu[1]].ravel()
    vals_delta_raw = (task - rest)[:, iu[0], iu[1]].ravel()

    # -------- Normalized values (z-space) --------
    vals_task_z  = z_transform_for_hist(vals_task_raw, corr_type)
    vals_rest_z  = z_transform_for_hist(vals_rest_raw, corr_type)
    # διαφορά στο z-space (όχι back-transform διαφοράς)
    vals_delta_z = vals_task_z - vals_rest_z

    def _hist(x, title, outpath, bins=60):
        x = x[np.isfinite(x)]
        plt.figure(figsize=(9, 5))
        plt.hist(x, bins=bins, density=True)
        plt.title(title); plt.xlabel("value"); plt.ylabel("Density")
        plt.tight_layout(); plt.savefig(outpath, dpi=300); plt.close()

    # --- Α. Histograms σε RAW space ---
    _hist(vals_task_raw,  f"{corr_type.upper()} — RAW distribution (Task)",
          os.path.join(figs_dir, f"dist_raw_{corr_type}_task.png"))
    _hist(vals_rest_raw,  f"{corr_type.upper()} — RAW distribution (Rest)",
          os.path.join(figs_dir, f"dist_raw_{corr_type}_rest.png"))
    _hist(vals_delta_raw, f"{corr_type.upper()} — RAW distribution Δ (Task−Rest)",
          os.path.join(figs_dir, f"dist_raw_{corr_type}_delta.png"))

    # --- Β. Histograms σε NORMALIZED (z) space ---
    _hist(vals_task_z,  f"{corr_type.upper()} — Normalized (z) distribution (Task)",
          os.path.join(figs_dir, f"dist_norm_{corr_type}_task.png"))
    _hist(vals_rest_z,  f"{corr_type.upper()} — Normalized (z) distribution (Rest)",
          os.path.join(figs_dir, f"dist_norm_{corr_type}_rest.png"))
    _hist(vals_delta_z, f"{corr_type.upper()} — Normalized (z) distribution Δz (Task−Rest)",
          os.path.join(figs_dir, f"dist_norm_{corr_type}_delta_z.png"))

    print(f"✅ Histograms (RAW & z) saved for {corr_type}")

    # --- 95% CI: στο z-space, με back-transform για ερμηνεία ---
    def ci_z(values_z: np.ndarray):
        x = values_z[np.isfinite(values_z)]
        n = x.size
        if n == 0: return np.nan, np.nan, np.nan, 0
        m, s = x.mean(), x.std(ddof=1)
        half = 1.96 * s / np.sqrt(n)
        return m, m - half, m + half, n

    # Task / Rest: CI στο z-space + back-transform στο original metric
    mt, lt, ht, nt = ci_z(vals_task_z)
    mr, lr, hr, nr = ci_z(vals_rest_z)

    # back-transform των ορίων (δεν ισούται με CI του μέσου raw, αλλά είναι standard report)
    mt_bt, lt_bt, ht_bt = inv_z(mt), inv_z(lt), inv_z(ht)
    mr_bt, lr_bt, hr_bt = inv_z(mr), inv_z(lr), inv_z(hr)

    # Δz: CI στο z-space (κρατάμε το z γιατί η διαφορά δεν έχει φυσικό back-transform)
    mdz, ldz, hdz, ndz = ci_z(vals_delta_z)

    # --- Αποθήκευση αναφοράς ---
    with open(os.path.join(stats_dir, f"grand_mean_ci_{corr_type}.txt"), "w", encoding="utf-8") as f:
        f.write(f"{corr_type.upper()} — S={len(used)} subjects, edges per subject={(N*(N-1))//2}\n")
        f.write("== Normalized (z) means & 95% CI ==\n")
        f.write(f"Task  mean(z): {mt:.4f} (95% CI {lt:.4f}, {ht:.4f})  [n={nt}]\n")
        f.write(f"Rest  mean(z): {mr:.4f} (95% CI {lr:.4f}, {hr:.4f})  [n={nr}]\n")
        f.write(f"Δz=Task−Rest mean: {mdz:.4f} (95% CI {ldz:.4f}, {hdz:.4f})  [n={ndz}]\n\n")
        f.write("== Back-transformed (tanh of mean z ± CI) for interpretability ==\n")
        f.write(f"Task  mean ρ≈ {mt_bt:.4f} (95% CI {lt_bt:.4f}, {ht_bt:.4f})\n")
        f.write(f"Rest  mean ρ≈ {mr_bt:.4f} (95% CI {lr_bt:.4f}, {hr_bt:.4f})\n")
        f.write("(Σημ.: το back-transform της Δz δεν ορίζεται ως «CI σε ρ».)\n")

    print(f"✅ Grand-mean 95% CI (z-space + back-transform) saved for {corr_type}")


# =========================
# 4C) Task–Rest ανά edge: paired t-test σε z(ρ) + FDR & Uncorrected
# =========================
def test_and_plot_task_rest_diff(corr_type: str) -> None:
    figure_dir = os.path.join(BASE_OUTPUT_DIR, corr_type, "figures")
    os.makedirs(figure_dir, exist_ok=True)

    task, rest, used = load_stacks(corr_type)
    if task is None:
        print(f"⚠️ Not enough valid data for heatmap ({corr_type})"); return

    S, N, _ = task.shape

    stat_mat = np.zeros((N, N))
    p_vals   = np.ones((N, N))

    for i in range(N):
        for j in range(N):
            x_task = task[:, i, j].astype(float)
            x_rest = rest[:, i, j].astype(float)
            # Μετασχηματισμός πριν το t-test
            z_task = z_transform_for_test(x_task, corr_type)
            z_rest = z_transform_for_test(x_rest, corr_type)
            t, p = ttest_rel(z_task, z_rest, nan_policy="omit")
            stat_mat[i, j] = 0.0 if not np.isfinite(t) else t
            p_vals[i, j]   = 1.0 if not np.isfinite(p) else p

    # Masks στο άνω τρίγωνο
    mask_ut = np.triu(np.ones_like(p_vals, dtype=bool), k=1)

    # FDR
    _, pvals_fdr_vec = fdrcorrection(p_vals[mask_ut], alpha=ALPHA_FDR)
    sig_mask_fdr = np.zeros_like(p_vals, dtype=bool)
    sig_mask_fdr[mask_ut] = pvals_fdr_vec < ALPHA_FDR
    sig_mask_fdr = sig_mask_fdr | sig_mask_fdr.T
    np.fill_diagonal(sig_mask_fdr, False)

    # Uncorrected
    sig_mask_unc = np.zeros_like(p_vals, dtype=bool)
    sig_mask_unc[mask_ut] = p_vals[mask_ut] < ALPHA_UNCORR
    sig_mask_unc = sig_mask_unc | sig_mask_unc.T
    np.fill_diagonal(sig_mask_unc, False)

    # Μέση διαφορά στο original space (Task-Rest)
    avg_task = np.nan_to_num(task.mean(axis=0), nan=0.0)
    avg_rest = np.nan_to_num(rest.mean(axis=0), nan=0.0)
    diff = avg_task - avg_rest

    def _save_outputs(mask, tag: str):
        sig_diff = np.zeros_like(diff)
        sig_diff[mask] = diff[mask]
        np.fill_diagonal(sig_diff, 0.0)

        # Heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(sig_diff, cmap="coolwarm", center=0, cbar_kws={'label': 'Δ Task-Rest'})
        plt.title(f"{corr_type.upper()} Δ(Task−Rest) — paired t-test on z(ρ) — {tag}")
        plt.xlabel("ROI j"); plt.ylabel("ROI i")
        plt.tight_layout()
        out_hm = os.path.join(figure_dir, f"group_{corr_type}_diff_heatmap_{tag}.png")
        plt.savefig(out_hm, dpi=300); plt.close()

        # Connectome
        try:
            atlas = datasets.fetch_atlas_schaefer_2018(n_rois=N_ROIS, resolution_mm=1)
            coords = plotting.find_parcellation_cut_coords(labels_img=atlas["maps"])
            fig = plotting.plot_connectome(
                sig_diff, coords,
                title=f"{corr_type.upper()} Connectome Δ(Task−Rest) — {tag}",
                edge_threshold="80%", node_size=20, colorbar=False
            )
            out_cn = os.path.join(figure_dir, f"group_{corr_type}_connectome_{tag}.png")
            fig.savefig(out_cn, dpi=300); plt.close()
            print(f"✅ Saved: {out_cn}")
        except Exception as e:
            print(f"⚠️ Connectome plot failed ({tag}): {e}")

        # CSV significant edges
        rows = []
        for i in range(N):
            for j in range(i+1, N):
                if mask[i, j]:
                    rows.append({
                        "ROI_i": i, "ROI_j": j,
                        "Network_i": roi_to_network(i),
                        "Network_j": roi_to_network(j),
                        "Delta_TaskMinusRest": float(diff[i, j]),
                        "p_value": float(p_vals[i, j])
                    })
        df_edges = pd.DataFrame(rows, columns=["ROI_i","ROI_j","Network_i","Network_j","Delta_TaskMinusRest","p_value"])
        out_edges = os.path.join(figure_dir, f"top_edges_{tag}_{corr_type}.csv")
        df_edges.to_csv(out_edges, index=False)
        print(f"✅ Saved: {out_hm}")
        print(f"✅ Saved: {out_edges}\n")

    # Παράγουμε ΚΑΙ τις δύο εκδοχές
    _save_outputs(sig_mask_fdr, f"fdr_q{ALPHA_FDR:.2f}")
    _save_outputs(sig_mask_unc, f"uncorr_p{ALPHA_UNCORR:.2f}")

# =========================
# 5) Pie chart από τα significant edges (per corr_type & tag)
# =========================
def plot_network_distribution_pie(corr_type: str, tag: str = None) -> None:
    if tag is None:
        tag = f"fdr_q{ALPHA_FDR:.2f}"
    edges_csv = os.path.join(
        BASE_OUTPUT_DIR, corr_type, "figures", f"top_edges_{tag}_{corr_type}.csv"
    )
    if not os.path.exists(edges_csv):
        print(f"⚠️ Cannot draw pie chart — CSV not found: {edges_csv}")
        return

    try:
        df = pd.read_csv(edges_csv)
    except pd.errors.EmptyDataError:
        print(f"⚠️ Pie skipped — empty CSV: {edges_csv}")
        return

    if df.empty or ("Network_i" not in df.columns) or ("Network_j" not in df.columns):
        print(f"⚠️ Pie skipped — no significant edges for {corr_type} ({tag}).")
        return

    networks = df["Network_i"].dropna().astype(str).tolist() + df["Network_j"].dropna().astype(str).tolist()
    counts = Counter(networks)
    if not counts:
        print(f"⚠️ Pie skipped — no networks to count for {corr_type} ({tag}).")
        return

    labels, sizes = zip(*sorted(counts.items(), key=lambda kv: kv[1], reverse=True))
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140)
    plt.title(f"{corr_type.upper()} — {tag}")
    plt.axis("equal")
    plt.tight_layout()
    out_pie = os.path.join(BASE_OUTPUT_DIR, corr_type, "figures", f"network_pie_{tag}_{corr_type}.png")
    plt.savefig(out_pie, dpi=300)
    plt.close()
    print(f"✅ Saved: {out_pie}")


# =========================
# Main
# =========================
if __name__ == "__main__":
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    for corr_type in CORR_TYPES:
        # Προαιρετικά descriptors
        # compute_and_save_metrics(corr_type)
        # plot_metric_comparison(corr_type)
        # plot_network_comparisons(corr_type)
        distributions_and_ci(corr_type)


        # Edge-level Task–Rest (both: FDR & uncorrected)
        # test_and_plot_task_rest_diff(corr_type)
        # plot_network_distribution_pie(corr_type, tag=f"fdr_q{ALPHA_FDR:.2f}")
        # plot_network_distribution_pie(corr_type, tag=f"uncorr_p{ALPHA_UNCORR:.2f}")

# %%
# Βοηθητικό snippet για να εκτυπώσεις σύνοψη pcca:

ALPHA_FDR=0.1
ALPHA_UNCORR=0.05
if True:
    ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    stats_dir = ROOT / "preprocessed data" / "correlation_results" / "pcca" / "stats"
    fig_dir   = ROOT / "preprocessed data" / "correlation_results" / "pcca" / "figures"
    print("stats_dir:", stats_dir)
    print("fig_dir:", fig_dir)
    for name in ["edges_task_vs_zero_pcca.csv", "edges_rest_vs_zero_pcca.csv"]:
        f = stats_dir / name
        if f.exists():
            df = pd.read_csv(f)
            n_fdr = int((df["p_fdr"] < ALPHA_FDR).sum())
            n_unc = int((df["p_raw"] < ALPHA_UNCORR).sum())
            print(f"{name}: FDR(q={ALPHA_FDR}) = {n_fdr} | uncorrected p<{ALPHA_UNCORR} = {n_unc}")
    f_unc = fig_dir / "top_edges_uncorr_p{:.2f}_pcca.csv".format(ALPHA_UNCORR)
    f_fdr = fig_dir / "top_edges_fdr_q{:.2f}_pcca.csv".format(ALPHA_FDR)
    if f_unc.exists(): print("Task–Rest, uncorrected edges:", len(pd.read_csv(f_unc)))
    if f_fdr.exists(): print("Task–Rest, FDR edges:", len(pd.read_csv(f_fdr)))
# %%
