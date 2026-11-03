from sklearn.preprocessing import OneHotEncoder
import pandas as pd

cities = pd.DataFrame({'city': ['Delhi', 'Mumbai', 'Delhi', 'Pune']})

ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
arr = ohe.fit_transform(cities)
print(ohe.get_feature_names_out())   # ['city_Mumbai' 'city_Pune']