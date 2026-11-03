
from sklearn.preprocessing import StandardScaler
import numpy as np

# Input dataset
X = np.array([
    [25, 30000],
    [32, 52000],
    [47, 90000],
    [51, 120000]
])

# Create StandardScaler object
sc = StandardScaler()

# Fit and transform the data
X_scaled = sc.fit_transform(X)

# Display results
print("Original Data:")
print(X)

print("\nScaled Data:")
print(X_scaled)

print("\nMean:")
print(sc.mean_)

print("\nStandard Deviation:")
print(sc.scale_)

print("\nVariance:")
print(sc.var_)
