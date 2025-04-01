import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn.maskers import NiftiLabelsMasker
from nilearn import datasets
from mvlearn.embed import GCCA

# === Step 1: Load the fMRI Data ===
fmri_files = [
    "sub-0001_task-restingstate_acq-seq_bold.nii.gz",
    "sub-0002_task-restingstate_acq-seq_bold.nii.gz",
    "sub-0003_task-restingstate_acq-seq_bold.nii.gz"
]

# === Step 2: Load a Brain Atlas for ROI Extraction ===
atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100)
masker = NiftiLabelsMasker(atlas.maps, standardize=True)

# === Step 3: Extract ROI-Based Time Series for Each Subject ===
fmri_time_series = []
for file in fmri_files:
    img = nib.load(file)  # Load NIfTI file
    time_series = masker.fit_transform(img)  # Extract ROI signals
    fmri_time_series.append(time_series)

# === Step 4: Remove Confounds (Highly Correlated ROIs) ===
n_confounds = 5  # Number of ROIs to remove
n_iterations = 10  # Number of iterations

fmri_cleaned = fmri_time_series.copy()

for iteration in range(n_iterations):
    print(f"Iteration {iteration + 1}/{n_iterations}")

    new_fmri_cleaned = []

    for X_k in fmri_cleaned:
        if np.any(np.isnan(X_k)):  
            print("Warning: NaN values found, replacing with zeros")
            X_k = np.nan_to_num(X_k)

        R_k = np.corrcoef(X_k, rowvar=False)  # (100, 100) correlation matrix

        # Identify the most correlated ROIs (excluding ROI 0 and 1)
        available_rois = np.setdiff1d(np.arange(X_k.shape[1]), [0, 1])
        sorted_indices = np.argsort(-np.abs(R_k), axis=1)[:, 1:n_confounds+2]

        top_rois = []
        for roi_list in sorted_indices:
            filtered_list = [roi for roi in roi_list if roi in available_rois]
            if len(filtered_list) >= n_confounds:
                top_rois.append(filtered_list[:n_confounds])
            else:
                top_rois.append([0] * n_confounds)  # Ensure fixed size

        top_rois = np.array(top_rois, dtype=int)  # Ensure integer indexing

        # Compute Z_k (Confound Matrix)
        Z_k = np.mean(X_k[:, top_rois], axis=1)  # (T, 5)

        # Partial Correlation Removal
        H_XZ = X_k.T @ Z_k
        H_ZZ_inv = np.linalg.pinv(Z_k.T @ Z_k)
        H_ZX = Z_k.T @ X_k
        X_k_partial = X_k - Z_k @ (H_ZZ_inv @ H_ZX)

        new_fmri_cleaned.append(X_k_partial)

    fmri_cleaned = new_fmri_cleaned  # Update data for the next iteration





# === Step 5: Apply Generalized Canonical Correlation Analysis (GCCA) ===
n_components = 10  # Number of common GCCA components

# Fit GCCA model on all subjects
gcca = GCCA(n_components=n_components)
gcca.fit(fmri_cleaned)
fmri_gcca = gcca.transform(fmri_cleaned)  # Transformed data in GCCA space


# === Step 6: Visualize the GCCA Components ===
fig, axes = plt.subplots(n_components, 1, figsize=(10, 8))
for i in range(n_components):
    for k in range(len(fmri_gcca)):  # For each subject
        axes[i].plot(fmri_gcca[k][:, i], label=f"GCCA Component {i+1} (Subject {k+1})")
    axes[i].legend()
    axes[i].set_ylabel("Activation")
    axes[i].set_xlabel("Time")
plt.tight_layout()
plt.show()
