from sklearn.preprocessing import MinMaxScaler
import numpy as np

X = np.array([
    [25, 30000],
    [32, 52000],
    [47, 90000],
    [51, 120000]
])
mm = MinMaxScaler(feature_range=(0, 1))
X_mm = mm.fit_transform(X)
print(mm.data_min_, mm.data_max_)