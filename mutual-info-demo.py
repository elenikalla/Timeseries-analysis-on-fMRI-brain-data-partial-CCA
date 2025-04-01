import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.stats import entropy

# Generate synthetic multivariate data
np.random.seed(42)
data_set1 = np.random.rand(100, 10)  # Sample data for variable X
data_set2 = np.random.rand(100, 10)  # Sample data for variable Y

# Function to calculate mutual information using KDE
def mutual_information_kde(x, y):
    # Fit KDE for each variable
    kde_x = KernelDensity(bandwidth=0.5)
    kde_x.fit(x)

    kde_y = KernelDensity(bandwidth=0.5)
    kde_y.fit(y)

    # Calculate individual PDFs
    pdf_x = np.exp(kde_x.score_samples(x))
    pdf_y = np.exp(kde_y.score_samples(y))

    # Calculate joint PDF
    pdf_joint = np.outer(pdf_x, pdf_y)

    # Calculate entropies and mutual information
    entropy_x = entropy(pdf_x)
    entropy_y = entropy(pdf_y)
    entropy_joint = entropy(pdf_joint.flatten())
    mutual_info = entropy_x + entropy_y - entropy_joint

    return mutual_info

# Calculate mutual information between the two sets using KDE
mi_score = mutual_information_kde(data_set1, data_set2)

print(f"Mutual Information Score: {mi_score}")