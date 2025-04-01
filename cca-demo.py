import numpy as np
from sklearn.cross_decomposition import CCA

# Generate synthetic fMRI data
np.random.seed(42)
fmri_data = np.random.rand(100, 10)  # 100 samples, 10 features

# Generate synthetic covariate data
covariate_data = np.random.rand(100, 5)  # 100 samples, 5 features

# Initialize CCA model
cca = CCA(n_components=2) 

# Fit the model
cca.fit(fmri_data, covariate_data)

# Transform the data into canonical variables
x_scores, y_scores = cca.transform(fmri_data, covariate_data)


import matplotlib.pyplot as plt

plt.scatter(x_scores[:, 0], x_scores[:, 1], label='X', c='blue')
plt.scatter(y_scores[:, 0], y_scores[:, 1], label='Y', c='red')
plt.xlabel('fMRI Data')
plt.ylabel('Covariate Data')
plt.legend()
plt.title('CCA Scatter Plot')
plt.show()




