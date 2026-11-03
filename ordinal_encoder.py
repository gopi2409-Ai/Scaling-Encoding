from sklearn.preprocessing import OrdinalEncoder
import pandas as pd

df = pd.DataFrame({'size': ['S', 'L', 'M', 'S'],
                   'edu':  ['BSc', 'PhD', 'MSc', 'BSc']})

oe = OrdinalEncoder(categories=[['S', 'M', 'L'], ['BSc', 'MSc', 'PhD']])
print(oe.fit_transform(df))