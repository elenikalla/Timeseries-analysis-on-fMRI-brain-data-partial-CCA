# %%
# %%
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.maskers import NiftiLabelsMasker
from nilearn import datasets, image

# === Define Subject IDs ===
subject_ids = [
    "sub-0001", "sub-0002", "sub-0003", "sub-0004", "sub-0005",
    "sub-0006", "sub-0007", "sub-0008", "sub-0009", "sub-0011"
]
# subject_ids = [
#     "sub-0001", "sub-0002"
# ]
# subject_ids = [
#     "sub-0001"
# ]

# %%
import nibabel as nib

# === Load preprocessed fMRI task data ===
img_task = nib.load("sub-0001_task-workingmemory_acq-seq_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")
data_task = img_task.get_fdata()  # Shape: (X, Y, Z, T)
header_task = img_task.header
print("Header of fMRI task data:", header_task.get_zooms())
print("Shape of fMRI task data:", data_task.shape)

# === Load preprocessed fMRI resting-state data ===
img_rest = nib.load("sub-0001_task-restingstate_acq-seq_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")
data_rest = img_rest.get_fdata()
header_rest = img_rest.header
print("Header of fMRI rest data:", header_rest.get_zooms())
print("Shape of fMRI rest data:", data_rest.shape)

# %%

fmri_rest_files = [
    f"{sid}_task-restingstate_acq-seq_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    for sid in subject_ids
]

fmri_task_files = [
    f"{sid}_task-workingmemory_acq-seq_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    for sid in subject_ids
]

tsv_files = [f"../{sid}.tsv" for sid in subject_ids]



# %%
# %%
# === Load Atlas and Define Masker ===
atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100)
masker = NiftiLabelsMasker(atlas.maps, standardize=True)

# %%
# %%
import numpy as np
import pandas as pd

def build_trial_wise_matrix(time_series_task, time_series_rest, tsv_path, TR=2.0, fixed_len=3):
    """
    Επιστρέφει αντίστοιχα blocks task και rest για κάθε trial, με βάση τα onset του .tsv.

    Parameters:
    - time_series_task: NumPy array (T_task, R)
    - time_series_rest: NumPy array (T_rest, R)
    - tsv_path: string, path προς .tsv αρχείο
    - TR: χρονική διάρκεια TR σε δευτερόλεπτα
    - fixed_len: αριθμός TRs ανά trial block (default: 3)

    Returns:
    - task_matrix: NumPy array (n_trials, fixed_len, n_rois)
    - rest_matrix: NumPy array (n_trials, fixed_len, n_rois)
    """
    df = pd.read_csv(tsv_path, sep="\t")
    n_timepoints_task, n_rois = time_series_task.shape
    n_timepoints_rest = time_series_rest.shape[0]

    task_matrix = []
    rest_matrix = []

    for _, row in df.iterrows():
        onset = row["onset"]
        start_tr = int(np.floor(onset / TR))
        end_tr = start_tr + fixed_len

        # Skip trial αν δεν χωράει ούτε σε task ούτε σε rest
        if end_tr > n_timepoints_task or end_tr > n_timepoints_rest:
            continue

        block_task = time_series_task[start_tr:end_tr]
        block_rest = time_series_rest[start_tr:end_tr]

        task_matrix.append(block_task)
        rest_matrix.append(block_rest)

    return np.array(task_matrix), np.array(rest_matrix)

# %%
# %%
subject_task_data = {}
subject_rest_data = {}

for sid, rest_file, task_file, tsv_file in zip(subject_ids, fmri_rest_files, fmri_task_files, tsv_files):
    print(f"Processing {sid}")

    img_task = nib.load(task_file)
    img_rest = nib.load(rest_file)

    ts_task = masker.fit_transform(img_task)
    ts_rest = masker.transform(img_rest)

    task_matrix, rest_matrix = build_trial_wise_matrix(ts_task, ts_rest, tsv_file)

    subject_task_data[sid] = task_matrix
    subject_rest_data[sid] = rest_matrix

# %%
for sid,sid in zip(subject_task_data,subject_rest_data):
    print(f"Subject {sid} task data shape",subject_task_data[sid].shape)
    print(f"Subject {sid} rest data shape",subject_rest_data[sid].shape)

# %%
import numpy as np

def compute_roi_correlation_matrix(data):
    """
    Υπολογίζει Pearson correlation matrix μεταξύ ROIs,
    flattening όλα τα TRs από όλα τα trials.

    Parameters:
    - data: NumPy array (n_trials, 3, n_rois)

    Returns:
    - corr_matrix: NumPy array (n_rois, n_rois)
    """
    n_trials, n_trs, n_rois = data.shape
    flat_data = data.reshape(-1, n_rois)  # shape: (n_trials * 3, n_rois)

    corr_matrix = np.corrcoef(flat_data.T)  # (n_rois, n_rois)
    return corr_matrix


# %%
task_corr_dict = {}
rest_corr_dict = {}

for sid in subject_ids:
    task_corr = compute_roi_correlation_matrix(subject_task_data[sid])
    rest_corr = compute_roi_correlation_matrix(subject_rest_data[sid])

    task_corr_dict[sid]=task_corr
    rest_corr_dict[sid]=rest_corr


for sid,sid in zip(task_corr_dict,rest_corr_dict):
    print(f"Subject {sid} task data correlation",task_corr_dict[sid].shape)
    print(f"Subject {sid} rest data correlation",task_corr_dict[sid].shape)


# %%
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA

def compute_partial_pearson_matrix(data, num_components=3):
    """
    Υπολογίζει πίνακα Partial Pearson Correlation μεταξύ ROIs,
    αφαιρώντας κοινή πληροφορία μέσω PCA (σε flatten δεδομένα από trials).

    Parameters:
    - data: NumPy array (n_trials, n_TRs, n_rois)
    - num_components: Αριθμός PCA components που θα αφαιρεθούν ως confounds

    Returns:
    - partial_corr_matrix: NumPy array (n_rois, n_rois) συμμετρικός
    """
    n_trials, n_trs, n_rois = data.shape
    flat_data = data.reshape(-1, n_rois)

    partial_corr_matrix = np.eye(n_rois)

    for i in range(n_rois):
        for j in range(i + 1, n_rois):
            X = flat_data[:, i].reshape(-1, 1)
            Y = flat_data[:, j].reshape(-1, 1)

            other_indices = [k for k in range(n_rois) if k != i and k != j]
            Z = flat_data[:, other_indices]

            num_components
            Z_pca = PCA(n_components=num_components).fit_transform(Z)

            def regress_out(A, Z_pca):
                reg = LinearRegression().fit(Z_pca, A)
                return A - reg.predict(Z_pca)

            X_resid = regress_out(X, Z_pca).flatten()
            Y_resid = regress_out(Y, Z_pca).flatten()

            corr = np.corrcoef(X_resid, Y_resid)[0, 1]
            partial_corr_matrix[i, j] = corr
            partial_corr_matrix[j, i] = corr

    return partial_corr_matrix

# %%
task_part_corr_dict = {}
rest_part_corr_dict = {}

for sid in subject_ids:
    task_part_corr = compute_partial_pearson_matrix(subject_task_data[sid])
    rest_part_corr = compute_partial_pearson_matrix(subject_rest_data[sid])

    task_part_corr_dict[sid]=task_part_corr
    rest_part_corr_dict[sid]=rest_part_corr
    print(f"Subject {sid} task data partial correlation",task_part_corr_dict[sid].shape)
    print(f"Subject {sid} rest data partial correlation",rest_part_corr_dict[sid].shape)


# %%
import numpy as np
from sklearn.cross_decomposition import CCA

def compute_cca_matrix_across_trials(data, n_components=1):
    """
    Υπολογίζει Canonical Correlation μεταξύ κάθε ζεύγους ROIs,
    χωρίς flatten — χρησιμοποιώντας τα 3 TRs ανά trial ως μεταβλητές.

    Parameters:
    - data: NumPy array (n_trials, 3, n_rois)
    - n_components: αριθμός CCA components (συνήθως 1)

    Returns:
    - cca_matrix: NumPy array (n_rois, n_rois), συμμετρικός
    """
    n_trials, n_trs, n_rois = data.shape
    cca_matrix = np.eye(n_rois)

    for i in range(n_rois):
        for j in range(i + 1, n_rois):
            X = data[:, :, i]  # shape: (n_trials, 3)
            Y = data[:, :, j]  # shape: (n_trials, 3)

            cca = CCA(n_components=n_components)
            X_c, Y_c = cca.fit_transform(X, Y)

            # Υπολογίζουμε την Pearson correlation του 1ου canonical pair
            corr = np.corrcoef(X_c[:, 0], Y_c[:, 0])[0, 1]
            cca_matrix[i, j] = corr
            cca_matrix[j, i] = corr

    return cca_matrix


# %%
task_cca_dict = {}
rest_cca_dict = {}

for sid in subject_ids:
    task_cca = compute_cca_matrix_across_trials(subject_task_data[sid])
    rest_cca = compute_cca_matrix_across_trials(subject_rest_data[sid])

    task_cca_dict[sid]=task_cca
    rest_cca_dict[sid]=rest_cca
    print(f"Subject {sid} task data cca",task_cca_dict[sid].shape)
    print(f"Subject {sid} rest data cca",task_cca_dict[sid].shape)

# %%
from scipy.linalg import eig
import numpy as np

def compute_pcca_per_trial_with_topkZ(trial_data, i, j, cca_matrix, k=2):
    """
    Computes Partial Canonical Correlation (PCCA) between ROI i and j
    in a single trial, using top-k confounding ROIs selected from cca_matrix.
    
    Parameters:
    - trial_data: np.array of shape (3, n_rois)
    - i, j: indices of X and Y
    - cca_matrix: np.array of shape (n_rois, n_rois), CCA values between ROIs
    - k: number of top Z ROIs to use for conditioning
    
    Returns:
    - rho: partial canonical correlation (scalar between 0 and 1)
    """
    n_rois = trial_data.shape[1]

    # === Extract X and Y ===
    X = trial_data[:, i].reshape(-1, 1)
    Y = trial_data[:, j].reshape(-1, 1)

    # === Select top-k Z ROIs based on relevance in cca_matrix ===
    candidate_indices = [z for z in range(n_rois) if z != i and z != j]
    relevance_scores = [max(cca_matrix[i, z], cca_matrix[j, z]) for z in candidate_indices]
    top_k_indices = [candidate_indices[z] for z in np.argsort(relevance_scores)[-k:]]
    Z = trial_data[:, top_k_indices]  # shape (3, k)

    def center(A): return A - A.mean(axis=0, keepdims=True)
    def cov(A, B): return center(A).T @ center(B) / (A.shape[0] - 1)

    try:
        # === Compute covariances ===
        Sxz = cov(X, Z)
        Syz = cov(Y, Z)
        Szz = cov(Z, Z)
        Sxy = cov(X, Y)
        Sxx = cov(X, X)
        Syy = cov(Y, Y)

        # === Compute inverse safely ===
        inv_Szz = np.linalg.pinv(Szz)

        # === Partial covariances ===
        Sxy_z = Sxy - Sxz @ inv_Szz @ Syz.T
        Syx_z = Sxy_z.T
        Sxx_z = Sxx - Sxz @ inv_Szz @ Sxz.T
        Syy_z = Syy - Syz @ inv_Szz @ Syz.T

        # === Check trace values before sqrt ===
        trace_x = np.trace(Sxx_z)
        trace_y = np.trace(Syy_z)

        if not np.isfinite(trace_x) or not np.isfinite(trace_y):
            return 0.0
        if trace_x <= 0 or trace_y <= 0:
            return 0.0

        norm_x = np.sqrt(trace_x)
        norm_y = np.sqrt(trace_y)

        # === Normalize ===
        Sxy_z_norm = Sxy_z / (norm_x * norm_y)
        Syx_z_norm = Sxy_z_norm.T

        A = np.block([
            [np.zeros_like(Sxy_z_norm), Sxy_z_norm],
            [Syx_z_norm, np.zeros_like(Syx_z_norm)]
        ])
        B = np.block([
            [np.eye(Sxx_z.shape[0]), np.zeros_like(Sxy_z_norm)],
            [np.zeros_like(Syx_z_norm), np.eye(Syy_z.shape[0])]
        ])

        eigvals, _ = eig(A, B)
        eigvals = np.real(eigvals)
        canonical_corrs = np.sort(np.abs(eigvals))[::-1]

        rho = canonical_corrs[0] if canonical_corrs.size > 0 else 0.0
        return np.clip(rho, 0, 1)

    except Exception:
        return 0.0

# %%
def compute_pcca_tensor(data, cca_matrix, k=2):
    """
    Computes the full PCCA tensor:
    shape = (n_trials, n_rois, n_rois)

    Parameters:
    - data: shape (n_trials, 3, n_rois)
    - cca_matrix: shape (n_rois, n_rois), from compute_cca_matrix_across_trials
    - k: number of top confounding ROIs to use in Z

    Returns:
    - pcca_tensor: (n_trials, n_rois, n_rois)
    """
    n_trials, _, n_rois = data.shape
    pcca_tensor = np.zeros((n_trials, n_rois, n_rois))

    for t in range(n_trials):
        trial = data[t]  # shape (3, n_rois)
        for i in range(n_rois):
            for j in range(i + 1, n_rois):
                pcca_val = compute_pcca_per_trial_with_topkZ(trial, i, j, cca_matrix, k)
                pcca_tensor[t, i, j] = pcca_val
                pcca_tensor[t, j, i] = pcca_val  

    return pcca_tensor

# %%
task_pcca_dict = {}
rest_pcca_dict = {}

for sid in subject_ids:
    task_pcca = compute_pcca_tensor(subject_task_data[sid],task_cca_dict[sid])
    rest_pcca = compute_pcca_tensor(subject_rest_data[sid],rest_cca_dict[sid])

    task_pcca_dict[sid]=task_pcca
    rest_pcca_dict[sid]=rest_pcca
    print(f"Subject {sid} task data pcca",task_pcca_dict[sid].shape)
    print(f"Subject {sid} rest data pcca",rest_pcca_dict[sid].shape)

# %%
import os
import pandas as pd

def save_all_correlations_to_csv(subject_ids, output_dir="correlation_csvs"):
    os.makedirs(output_dir, exist_ok=True)

    for sid in subject_ids:
        print(f"Saving correlations for {sid}...")

        matrices = {
            "task_pearson": task_corr_dict[sid],
            "rest_pearson": rest_corr_dict[sid],
            "task_partial": task_part_corr_dict[sid],
            "rest_partial": rest_part_corr_dict[sid],
            "task_cca": task_cca_dict[sid],
            "rest_cca": rest_cca_dict[sid]
        }

        for name, matrix in matrices.items():
            df = pd.DataFrame(matrix)
            df.to_csv(os.path.join(output_dir, f"{sid}_{name}.csv"), index=False)

    print(f"All CSVs saved in: {os.path.abspath(output_dir)}")


save_all_correlations_to_csv(subject_ids)

# %%
import pandas as pd
import os

def save_pcca_3d_to_csv(pcca_tensor, sid, condition, output_dir="correlation_csvs"):
    """
    Saves a (n_trials, n_rois, n_rois) PCCA tensor to a long-format CSV file.

    Parameters:
        pcca_tensor: np.array (n_trials, n_rois, n_rois)
        sid: subject ID (e.g. "sub-0001")
        condition: "task" or "rest"
        output_dir: folder to save the CSV
    """
    os.makedirs(output_dir, exist_ok=True)
    n_trials, n_rois, _ = pcca_tensor.shape

    records = []
    for t in range(n_trials):
        for i in range(n_rois):
            for j in range(n_rois):
                records.append({
                    "trial": t,
                    "roi_i": i,
                    "roi_j": j,
                    "pcca_value": pcca_tensor[t, i, j]
                })

    df = pd.DataFrame(records)
    df.to_csv(f"{output_dir}/{sid}_{condition}_pcca_3d.csv", index=False)
    print(f"✅ Saved: {output_dir}/{sid}_{condition}_pcca_3d.csv")
# %%
for sid in subject_ids:
    save_pcca_3d_to_csv(task_pcca_dict[sid], sid, "task")
    save_pcca_3d_to_csv(rest_pcca_dict[sid], sid, "rest")



