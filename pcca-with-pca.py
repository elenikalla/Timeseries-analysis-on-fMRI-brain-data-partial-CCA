import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import CCA
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn.maskers import NiftiLabelsMasker
from nilearn import datasets

# === Step 1: Load the fMRI Data ===
fmri_rest_files = [
    "sub-0001_task-restingstate_acq-seq_bold.nii.gz",
    "sub-0002_task-restingstate_acq-seq_bold.nii.gz",
    "sub-0003_task-restingstate_acq-seq_bold.nii.gz",
    "sub-0004_task-restingstate_acq-seq_bold.nii.gz",
    "sub-0005_task-restingstate_acq-seq_bold.nii.gz",
    "sub-0006_task-restingstate_acq-seq_bold.nii.gz",
    "sub-0007_task-restingstate_acq-seq_bold.nii.gz",
    "sub-0008_task-restingstate_acq-seq_bold.nii.gz",
    "sub-0009_task-restingstate_acq-seq_bold.nii.gz",
    "sub-0011_task-restingstate_acq-seq_bold.nii.gz",
]

fmri_task_files = [
    "sub-0001_task-workingmemory_acq-seq_bold.nii.gz",
    "sub-0002_task-workingmemory_acq-seq_bold.nii.gz",
    "sub-0003_task-workingmemory_acq-seq_bold.nii.gz",
    "sub-0004_task-workingmemory_acq-seq_bold.nii.gz",
    "sub-0005_task-workingmemory_acq-seq_bold.nii.gz",
    "sub-0006_task-workingmemory_acq-seq_bold.nii.gz",
    "sub-0007_task-workingmemory_acq-seq_bold.nii.gz",
    "sub-0008_task-workingmemory_acq-seq_bold.nii.gz",
    "sub-0009_task-workingmemory_acq-seq_bold.nii.gz",
    "sub-0011_task-workingmemory_acq-seq_bold.nii.gz"
]

tsv_files = [
    "sub-01.tsv",
    "sub-02.tsv",
    "sub-03.tsv",
    "sub-04.tsv",  
    "sub-05.tsv",
    "sub-06.tsv",
    "sub-07.tsv",
    "sub-08.tsv",
    "sub-09.tsv",
    "sub-11.tsv"
]


# === Step 2: Load a Brain Atlas for ROI Extraction ===
atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100)
masker = NiftiLabelsMasker(atlas.maps, standardize=True)

# === Step 3: Extract ROI-Based Time Series for Each Subject ===
fmri_rest_time_series = []
for file in fmri_rest_files:
    img = nib.load(file)  # Load NIfTI file
    time_series = masker.fit_transform(img)  # Extract ROI signals
    fmri_rest_time_series.append(time_series)

# === Step 4: Extract ROI-Based Task Time Series for Each Subject ===
fmri_task_time_series = []
for file in fmri_task_files:
    img = nib.load(file)  # Load NIfTI file
    time_series = masker.fit_transform(img)  # Extract ROI signals
    fmri_task_time_series.append(time_series)
    
# === Load fMRI Data for 1 Subject ===
subject_data = fmri_rest_time_series[0]  # (timepoints x ROIs)
num_rois = subject_data.shape[1]  # Number of ROIs

# === Step 1: Iterate Over All Pairs of ROIs ===
pcca_results = np.zeros((num_rois, num_rois))

for i in range(num_rois):
    for j in range(i + 1, num_rois):  # Avoid duplicate pairs
        X = subject_data[:, i].reshape(-1, 1)  # ROI i time series (timepoints x 1)
        Y = subject_data[:, j].reshape(-1, 1)  # ROI j time series (timepoints x 1)

        # === Step 2: Select Remaining ROIs as Z ===
        remaining_indices = [idx for idx in range(num_rois) if idx != i and idx != j]
        Z = subject_data[:, remaining_indices]  # (timepoints x (num_rois - 2))

        # === Step 3: Apply PCA on Z to Reduce to 3 Components ===
        pca = PCA(n_components=3)  # Reduce all remaining ROIs to 3 components
        Z_pca = pca.fit_transform(Z)  # (timepoints x 3)

        # === Step 4: Remove the Effect of Z_pca from X and Y ===
        def remove_confound(X, Z_pca):
            """Αφαιρεί την επίδραση του Z_pca από το X μέσω γραμμικής παλινδρόμησης"""
            reg = LinearRegression().fit(Z_pca, X)  # Fit regression model
            return X - reg.predict(Z_pca)  # Compute residuals

        X_residual = remove_confound(X, Z_pca)  # Καθαρό X
        Y_residual = remove_confound(Y, Z_pca)  # Καθαρό Y

        # === Step 5: Apply CCA on Residuals ===
        cca = CCA(n_components=1)  # 1st canonical correlation
        X_c, Y_c = cca.fit_transform(X_residual, Y_residual)

        # === Step 6: Compute Partial Canonical Correlation ===
        pcca_value = np.corrcoef(X_c.T, Y_c.T)[0, 1]
        pcca_results[i, j] = pcca_value
        pcca_results[j, i] = pcca_value  # Symmetric matrix

# === Display Top 10 PCCA Correlations ===
sorted_pcca = sorted(
    [(i, j, pcca_results[i, j]) for i in range(num_rois) for j in range(i + 1, num_rois)],
    key=lambda x: -abs(x[2])
)[:10]

print("\nTop 10 Partial Canonical Correlations (PCCA) Between ROIs:")
for i, j, corr in sorted_pcca:
    print(f"ROI {i} <-> ROI {j} | PCCA: {corr:.3f}")

import numpy as np
import matplotlib.pyplot as plt

def get_significant_pcca(pcca_results, method="percentile", threshold=95):
    """
    Εντοπίζει τις πιο σημαντικές συσχετίσεις PCCA βάσει κατωφλίου.

    Args:
        pcca_results (np.ndarray): Πίνακας (num_rois x num_rois) με τις PCCA τιμές.
        method (str): Τρόπος επιλογής threshold ("percentile", "z-score" ή "fixed").
        threshold (float): Αν "percentile", π.χ. 95 → κρατά τις κορυφαίες 5% τιμές.
                           Αν "fixed", σταθερή τιμή κατωφλίου.
                           Αν "z-score", διατηρεί τιμές πάνω από mean + threshold * std.

    Returns:
        list: Λίστα με tuples (ROI1, ROI2, PCCA).
    """
    num_rois = pcca_results.shape[0]
    pcca_values = []

    for i in range(num_rois):
        for j in range(i + 1, num_rois):  # Avoid duplicate pairs
            pcca_values.append((i, j, pcca_results[i, j]))

    pcca_values = np.array(pcca_values, dtype=[("roi1", int), ("roi2", int), ("pcca", float)])

    if method == "percentile":
        percentile_value = np.percentile(pcca_values["pcca"], threshold)
        significant_pcca = pcca_values[pcca_values["pcca"] >= percentile_value]
    elif method == "z-score":
        mean_pcca = np.mean(pcca_values["pcca"])
        std_pcca = np.std(pcca_values["pcca"])
        significant_pcca = pcca_values[pcca_values["pcca"] >= mean_pcca + threshold * std_pcca]
    elif method == "fixed":
        significant_pcca = pcca_values[pcca_values["pcca"] >= threshold]
    else:
        raise ValueError("Invalid method. Choose 'percentile', 'z-score', or 'fixed'.")

    return significant_pcca

def plot_significant_pcca(pcca_results, significant_pcca):
    """
    Δημιουργεί ένα scatter plot με τις σημαντικές συσχετίσεις PCCA.

    Args:
        pcca_results (np.ndarray): Συμμετρικός πίνακας (num_rois x num_rois) με τις PCCA τιμές.
        significant_pcca (list): Λίστα με tuples (roi1, roi2, pcca_value) που έχουν ξεπεράσει το threshold.
    """
    num_rois = pcca_results.shape[0]

    # === Prepare Data for Plot ===
    roi1_list = [roi1 for roi1, roi2, pcca in significant_pcca]
    roi2_list = [roi2 for roi1, roi2, pcca in significant_pcca]
    pcca_values = [pcca for roi1, roi2, pcca in significant_pcca]

    # === Create Scatter Plot ===
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(roi1_list, roi2_list, c=pcca_values, cmap="coolwarm", edgecolors="black", alpha=0.75)
    plt.colorbar(scatter, label="Partial Canonical Correlation (PCCA)")
    
    # Labels and Styling
    plt.xlabel("ROI 1")
    plt.ylabel("ROI 2")
    plt.title("Significant PCCA Correlations Between ROIs")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xticks(np.arange(0, num_rois, step=10))  
    plt.yticks(np.arange(0, num_rois, step=10))

    # Show Plot
    plt.show()

# === Example Usage (Using Your Computed pcca_results) ===
significant_pcca = get_significant_pcca(pcca_results, method="percentile", threshold=95)
plot_significant_pcca(pcca_results, significant_pcca)

# # === Display Results ===
# print("\nTop Significant PCCA Correlations:")
# for roi1, roi2, pcca in top_pcca:a
#     print(f"ROI {roi1} <-> ROI {roi2} | PCCA: {pcca:.3f}")


drgdrfgdfg