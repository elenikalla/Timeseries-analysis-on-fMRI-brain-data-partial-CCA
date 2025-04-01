import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn.maskers import NiftiLabelsMasker
from nilearn import datasets
from sklearn.decomposition import NMF
from mvlearn.embed import GCCA
import seaborn as sns

# Step 1: Load the fMRI Data
fmri_files = [
    "sub-0001_task-restingstate_acq-seq_bold.nii.gz",
    "sub-0002_task-restingstate_acq-seq_bold.nii.gz",
    "sub-0003_task-restingstate_acq-seq_bold.nii.gz"
]

# Step 2: Load a Brain Atlas for ROI Extraction
atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100)
masker = NiftiLabelsMasker(atlas.maps, standardize=True)

# Step 3: Extract ROI-Based Time Series from Each Subject
fmri_time_series = []
for file in fmri_files:
    img = nib.load(file)  # Load NIfTI file
    time_series = masker.fit_transform(img)  # Extract ROI signals
    fmri_time_series.append(time_series)

# Step 4: Apply GCCA for Common Spatial Subspace
n_components = 10  # Estimated number of shared spatial components
gcca = GCCA(n_components=n_components)
X_gcca = gcca.fit_transform(fmri_time_series)

# Step 5: Solve Semi-Orthogonal Nonnegative Matrix Factorization (s-ONMF)
# Shift data to ensure non-negative values
X_shifted = X_gcca[0] - X_gcca[0].min()
nmf = NMF(n_components=n_components, init='random', solver='mu', max_iter=500)
A = nmf.fit_transform(X_shifted)  # Common spatial factor (Brain parcellation)
B = nmf.components_.T  # Coefficient matrix

# Step 6: Extract Subject-Specific Time Series
subject_time_series = [X_gcca[k] @ B for k in range(len(fmri_time_series))]

# Step 7: Plot GCCA Components
fig, axes = plt.subplots(n_components, 1, figsize=(10, 8))
for i in range(n_components):
    axes[i].plot(subject_time_series[0][:, i], label=f"Component {i+1}")
    axes[i].legend()
    axes[i].set_ylabel("Activation")
    axes[i].set_xlabel("Time")
plt.tight_layout()
plt.show()

# Step 8: Visualize Brain Parcellation
plt.figure(figsize=(10, 6))
sns.heatmap(A, cmap="viridis", cbar=True)
plt.xlabel("Parcellation Components")
plt.ylabel("Brain Regions (ROIs)")
plt.title("Brain Parcellation via NMF")
plt.show()

import numpy as np
from scipy.stats import pearsonr

n_components = 10  # Αριθμός GCCA συνιστωσών
n_subjects = len(X_gcca)  # Αριθμός υποκειμένων

# Δημιουργούμε έναν πίνακα για τις συσχετίσεις
gcca_correlations = np.zeros((n_subjects, n_subjects, n_components))

for i in range(n_subjects):
    for j in range(n_subjects):
        if i != j:  # Δεν συγκρίνουμε ένα υποκείμενο με τον εαυτό του
            for component in range(n_components):
                gcca_correlations[i, j, component] = pearsonr(
                    X_gcca[i][:, component], X_gcca[j][:, component]
                )[0]

# Υπολογίζουμε τον μέσο όρο των συσχετίσεων για κάθε GCCA συνιστώσα
mean_correlations = np.mean(gcca_correlations, axis=(0, 1))

print("Μέσες συσχετίσεις GCCA συνιστωσών μεταξύ των υποκειμένων:\n", mean_correlations)
