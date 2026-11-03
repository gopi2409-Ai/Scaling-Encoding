from sklearn.preprocessing import PowerTransformer, QuantileTransformer
import numpy as np

# Input dataset
X = np.array([
    [25, 30000],
    [32, 52000],
    [47, 90000],
    [51, 120000]
])

# Makes skewed data more Gaussian
pt = PowerTransformer(method='yeo-johnson')   # handles negatives; 'box-cox' needs positives
X_pt = pt.fit_transform(X)
print(X_pt)

# Maps data to uniform or normal distribution via ranks
qt = QuantileTransformer(output_distribution='normal', n_quantiles=4)
X_qt = qt.fit_transform(X)
print(X_qt)