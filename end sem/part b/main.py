import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Read dataset name/path from input
file_path = input().strip()

# Load the dataset
data = pd.read_csv(file_path)

# Select only numerical columns
X = data.select_dtypes(include=['float64', 'int64'])

# Standardize the dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA with 2 components
pca = PCA(n_components=2)
pca.fit(X_scaled)

# Print explained variance ratio
print("Explained Variance Ratio:", pca.explained_variance_ratio_)
