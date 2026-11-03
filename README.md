## Feature Engineering: Scaling & Encoding

## Overview
Today I learned about feature scaling and encoding using Python and Scikit-learn.

These techniques help prepare data before training a machine learning model.

## Topics Covered

### 1. Feature Scaling
Feature scaling brings numerical values into a similar range.

Methods learned:
- StandardScaler
- MinMaxScaler
- MaxAbsScaler
- RobustScaler
- Normalizer
- PowerTransformer
- QuantileTransformer

### 2. Categorical Encoding
Encoding converts text values into numerical values.

Methods learned:
- LabelEncoder
- OrdinalEncoder
- OneHotEncoder
- TargetEncoder

### 3. Data Preprocessing
Learned how to use:
- SimpleImputer – handles missing values.
- ColumnTransformer – applies different transformations to different columns.
- Pipeline – combines preprocessing and machine learning steps.

### 4. Machine Learning Models
Practised using:
- K-Nearest Neighbors (KNN)
- Logistic Regression
- Random Forest

Also learned about:
- Train and Test Split
- Cross Validation
- ROC-AUC Score
- Classification Report
- Data Leakage

## Mini Project – Customer Churn Prediction

### Project Description
Created a sample telecom customer dataset to predict whether a customer will leave the company.

The dataset contains:
- Age
- Monthly Charge
- Annual Income
- Tenure Months
- City
- Plan
- Paperless Billing
- Churn

### Project Steps

1. Created a dataset with 1000 customer records.
2. Handled missing values.
3. Split the dataset into training and testing data.
4. Applied StandardScaler, MinMaxScaler and RobustScaler.
5. Used OneHotEncoder and OrdinalEncoder.
6. Built preprocessing using ColumnTransformer and Pipeline.
7. Compared KNN, Logistic Regression and Random Forest.
8. Evaluated the final model using classification report and ROC-AUC.
9. Tested the model with a new customer and an unseen city.
10. Saved and loaded the trained model using Joblib.

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Joblib

## Libraries Installation

Run the following command:

```bash
pip install numpy pandas scikit-learn joblib
```

## How to Run

```bash
python customer_churn_pipeline.py
```

## Model Saving

The trained model is saved as:

```text
churn_pipeline.joblib
```

This saved model can be loaded later to make predictions without training it again.

## What I Learned

- Why feature scaling is important.
- Difference between StandardScaler and MinMaxScaler.
- How to handle outliers using RobustScaler.
- How to convert categorical data into numerical values.
- How to handle missing values.
- How to build a complete machine learning pipeline.
- Why data leakage should be avoided.
- How to evaluate and save a machine learning model.

## Conclusion

This practice helped me understand how to prepare real-world data for machine learning and build a complete customer churn prediction pipeline using Scikit-learn.
