
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split

# 1. Create sample dataset
data = {
    'age': [25, 30, 35, 40, 45, 28, 32, 50, 38, 29, 42, 48],
    'salary': [30000, 40000, 50000, 60000, 80000, 35000,
               45000, 100000, 65000, 38000, 75000, 95000],
    'city': ['Bangalore', 'Chennai', 'Hyderabad', 'Bangalore',
             'Chennai', 'Hyderabad', 'Bangalore', 'Chennai',
             'Hyderabad', 'Bangalore', 'Chennai', 'Hyderabad'],
    'education': ['UG', 'PG', 'UG', 'PhD', 'PG', 'School',
                  'UG', 'PhD', 'PG', 'School', 'PhD', 'PG'],
    'purchased': [0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1]
}

df = pd.DataFrame(data)

# 2. Separate features and target
X = df[['age', 'salary', 'city', 'education']]
y = df['purchased']

# 3. Define columns
num_cols = ['age', 'salary']
cat_nom = ['city']
cat_ord = ['education']

# 4. Numerical pipeline
num_pipe = Pipeline([
    ('imp', SimpleImputer(strategy='median')),
    ('sc', StandardScaler())
])

# 5. Nominal categorical pipeline
nom_pipe = Pipeline([
    ('imp', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore'))
])

# 6. Ordinal categorical pipeline
ord_pipe = Pipeline([
    ('imp', SimpleImputer(strategy='most_frequent')),
    ('ord', OrdinalEncoder(
        categories=[['School', 'UG', 'PG', 'PhD']]
    ))
])

# 7. Combine preprocessing
pre = ColumnTransformer([
    ('num', num_pipe, num_cols),
    ('nom', nom_pipe, cat_nom),
    ('ord', ord_pipe, cat_ord)
], remainder='drop')

# 8. Create complete ML pipeline
model = Pipeline([
    ('pre', pre),
    ('clf', LogisticRegression(max_iter=1000))
])

# 9. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# 10. Train model
model.fit(X_train, y_train)

# 11. Make predictions
y_pred = model.predict(X_test)

# 12. Display results
print("Training completed successfully!")

print("\nActual Values:")
print(y_test.values)

print("\nPredicted Values:")
print(y_pred)

print("\nModel Accuracy:")
print(model.score(X_test, y_test))

# 13. Predict for a new customer
new_customer = pd.DataFrame({
    'age': [33],
    'salary': [55000],
    'city': ['Bangalore'],
    'education': ['PG']
})

prediction = model.predict(new_customer)

print("\nNew Customer Prediction:")

if prediction[0] == 1:
    print("Customer is likely to purchase.")
else:
    print("Customer is unlikely to purchase.")
