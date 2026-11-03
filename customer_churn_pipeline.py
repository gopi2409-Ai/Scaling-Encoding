import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    OneHotEncoder,
    OrdinalEncoder,
    FunctionTransformer
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score, StratifiedKFold

from sklearn.metrics import classification_report, roc_auc_score


# ============================================================
# STEP 1: CREATE DATASET
# ============================================================

print("\n========== STEP 1: CREATE DATASET ==========")

rng = np.random.default_rng(42)
n = 1000

df = pd.DataFrame({
    'age': rng.integers(18, 70, n),
    'monthly_charge': rng.normal(70, 25, n).clip(10, 200).round(2),
    'annual_income': rng.lognormal(12.5, 0.6, n).round(0),
    'tenure_months': rng.integers(1, 72, n),
    'city': rng.choice(
        ['Delhi', 'Mumbai', 'Bengaluru', 'Pune', 'Chennai'], n
    ),
    'plan': rng.choice(
        ['Basic', 'Standard', 'Premium'],
        n,
        p=[0.5, 0.3, 0.2]
    ),
    'paperless': rng.choice(['Yes', 'No'], n)
})

# Generate churn probability
logit = (
    -1.5
    + 0.025 * (df['monthly_charge'] - 70)
    - 0.04 * df['tenure_months']
    + 0.8 * (df['plan'] == 'Basic')
    + 0.4 * (df['paperless'] == 'Yes')
    + 0.5 * (df['city'] == 'Mumbai')
)

df['churn'] = (
    rng.random(n) < 1 / (1 + np.exp(-logit))
).astype(int)

# Inject missing values
df.loc[rng.choice(n, 30, replace=False), 'monthly_charge'] = np.nan
df.loc[rng.choice(n, 20, replace=False), 'city'] = np.nan

print("\nFirst five records:")
print(df.head())

print("\nDataset description:")
print(df.describe())

print("\nChurn distribution:")
print(df['churn'].value_counts(normalize=True))


# ============================================================
# STEP 2: TRAIN / TEST SPLIT
# ============================================================

print("\n========== STEP 2: TRAIN TEST SPLIT ==========")

X = df.drop(columns='churn')
y = df['churn']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)


# ============================================================
# STEP 3: MANUAL SCALING
# ============================================================

print("\n========== STEP 3: MANUAL SCALING ==========")

num_cols = [
    'age',
    'monthly_charge',
    'annual_income',
    'tenure_months'
]

num_train = X_train[num_cols].fillna(
    X_train[num_cols].median()
)

scalers = [
    ('Standard', StandardScaler()),
    ('MinMax', MinMaxScaler()),
    ('Robust', RobustScaler())
]

for name, scaler in scalers:

    arr = scaler.fit_transform(num_train)

    print(f"\n--- {name} Scaler ---")

    scaled_df = pd.DataFrame(arr, columns=num_cols)

    print(
        scaled_df.describe()
        .loc[['mean', 'std', 'min', 'max']]
        .round(2)
    )


# ============================================================
# STEP 3B: ENCODERS
# ============================================================

print("\n========== STEP 3B: ENCODERS ==========")

ohe = OneHotEncoder(
    handle_unknown='ignore',
    sparse_output=False
)

city_enc = ohe.fit_transform(
    X_train[['city']].fillna('Unknown')
)

print("\nCity encoded feature names:")
print(ohe.get_feature_names_out())

ordenc = OrdinalEncoder(
    categories=[['Basic', 'Standard', 'Premium']]
)

plan_enc = ordenc.fit_transform(
    X_train[['plan']]
)

print("\nPlan encoded values:")
print(plan_enc[:5].ravel())


# ============================================================
# STEP 4: BUILD COLUMN TRANSFORMER
# ============================================================

print("\n========== STEP 4: COLUMN TRANSFORMER ==========")

def make_preprocessor(scaler):

    num_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', scaler)
    ])

    nominal_pipe = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    ordinal_pipe = Pipeline([
        ('ordinal', OrdinalEncoder(
            categories=[['Basic', 'Standard', 'Premium']]
        ))
    ])

    binary_pipe = Pipeline([
        ('onehot', OneHotEncoder(drop='if_binary'))
    ])

    return ColumnTransformer([
        ('num', num_pipe, num_cols),
        ('nominal', nominal_pipe, ['city']),
        ('ordinal', ordinal_pipe, ['plan']),
        ('binary', binary_pipe, ['paperless'])
    ])


print("ColumnTransformer created successfully!")


# ============================================================
# STEP 5: COMPARE MODELS AND SCALERS
# ============================================================

print("\n========== STEP 5: MODEL COMPARISON ==========")

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scaler_options = {
    'No scaling': FunctionTransformer(),
    'Standard': StandardScaler(),
    'MinMax': MinMaxScaler(),
    'Robust': RobustScaler()
}

models = {
    'KNN': KNeighborsClassifier(n_neighbors=15),

    'LogisticReg': LogisticRegression(
        max_iter=2000
    ),

    'RandomForest': RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
}

results = {}

for s_name, scaler in scaler_options.items():

    for m_name, model in models.items():

        pipe = Pipeline([
            ('pre', make_preprocessor(scaler)),
            ('clf', model)
        ])

        scores = cross_val_score(
            pipe,
            X_train,
            y_train,
            cv=cv,
            scoring='roc_auc'
        )

        results.setdefault(m_name, {})[s_name] = scores.mean()

        print(
            f"{m_name} | {s_name} | "
            f"ROC-AUC: {scores.mean():.3f}"
        )

print("\nFinal Comparison Table:")

print(pd.DataFrame(results).round(3))


# ============================================================
# STEP 6: DATA LEAKAGE DEMONSTRATION
# ============================================================

print("\n========== STEP 6: DATA LEAKAGE ==========")

# WRONG APPROACH:
# Fitting scaler on the complete dataset before splitting
wrong_scaler = StandardScaler()

wrong_scaler.fit(
    df[num_cols].fillna(0)
)

print("Wrong approach: Scaler has seen the full dataset.")

# RIGHT APPROACH:
# Pipeline fits preprocessing using training data only.
print("Right approach: Pipeline learns from training folds only.")


# ============================================================
# STEP 7: FINAL MODEL AND EVALUATION
# ============================================================

print("\n========== STEP 7: FINAL MODEL ==========")

final = Pipeline([
    ('pre', make_preprocessor(StandardScaler())),

    ('clf', LogisticRegression(max_iter=2000))
])

# Train final model
final.fit(X_train, y_train)

# Predictions
pred = final.predict(X_test)

proba = final.predict_proba(X_test)[:, 1]

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        pred,
        zero_division=0
    )
)

print(
    "ROC-AUC:",
    round(roc_auc_score(y_test, proba), 3)
)


# Inspect engineered features and coefficients
print("\n========== FEATURE COEFFICIENTS ==========")

names = final.named_steps['pre'].get_feature_names_out()

coefs = final.named_steps['clf'].coef_[0]

coef_series = pd.Series(
    coefs,
    index=names
).sort_values()

print(coef_series)


# ============================================================
# STEP 8: TEST UNSEEN CATEGORY AND SAVE MODEL
# ============================================================

print("\n========== STEP 8: TEST NEW CUSTOMER ==========")

new_customer = pd.DataFrame([{
    'age': 35,
    'monthly_charge': 90.0,
    'annual_income': 450000,
    'tenure_months': 4,
    'city': 'Kolkata',
    'plan': 'Basic',
    'paperless': 'Yes'
}])

prediction_probability = final.predict_proba(new_customer)

print("\nNew Customer Churn Probability:")

print(prediction_probability)

prediction = final.predict(new_customer)

if prediction[0] == 1:
    print("Prediction: Customer may churn.")
else:
    print("Prediction: Customer may not churn.")


# Save trained model
joblib.dump(final, 'churn_pipeline.joblib')

print("\nModel saved successfully!")


# Load saved model
loaded = joblib.load('churn_pipeline.joblib')

print("Saved model loaded successfully!")

# Test loaded model
print(
    "\nLoaded Model Prediction:",
    loaded.predict(new_customer)
)

print("\n========== PROJECT COMPLETED ==========")
