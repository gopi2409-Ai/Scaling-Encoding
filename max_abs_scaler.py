from sklearn.preprocessing import MaxAbsScaler
import numpy as np

# Input dataset
X = np.array([
    [25, 30000],
    [32, 52000],
    [47, 90000],
    [51, 120000]
])

X_ma = MaxAbsScaler().fit_transform(X)
print(X_ma)
