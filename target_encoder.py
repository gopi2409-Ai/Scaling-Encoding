from sklearn.preprocessing import TargetEncoder
import pandas as pd

X = pd.DataFrame({'zip': ['A', 'A', 'B', 'B', 'C', 'C', 'A', 'B']})
y = [1, 0, 1, 1, 0, 0, 1, 0]

te = TargetEncoder(smooth='auto', cv=2, random_state=0)
print(te.fit_transform(X, y))