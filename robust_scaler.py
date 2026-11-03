from sklearn.preprocessing import RobustScaler
import numpy as np

# Input dataset
X = np.array([
    [25, 30000],
    [32, 52000],
    [47, 90000],
    [51, 120000]
])

rs = RobustScaler(quantile_range=(25.0, 75.0))
X_rs = rs.fit_transform(X)
print(rs.center_, rs.scale_)   # median and IQR