from sklearn.preprocessing import Normalizer
import numpy as np

# Input dataset
X = np.array([
    [25, 30000],
    [32, 52000],
    [47, 90000],
    [51, 120000]
])
norm = Normalizer(norm='l2')   # 'l1', 'l2' or 'max'
X_n = norm.fit_transform(X)
print(X_n)