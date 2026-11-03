from sklearn.preprocessing import LabelEncoder

y = ['cat', 'dog', 'dog', 'bird']
le = LabelEncoder()
y_enc = le.fit_transform(y)     # [1, 2, 2, 0]
print(le.classes_)              # ['bird' 'cat' 'dog']
print(le.inverse_transform([0, 2]))