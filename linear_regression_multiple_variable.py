# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 09:09:20 2026

@author: I3D-MSI
"""


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

import statsmodels.api as sm
import matplotlib.pyplot as plt




file_path = "mlr_housing.csv"

df = pd.read_csv(file_path)

print("\n================ DATASET =================")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())



print("\n================ DATA INFORMATION =================")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())



# Variables containing yes/no values
binary_columns = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea"
]

# Convert yes/no into 1/0
for col in binary_columns:
    df[col] = df[col].map({"yes": 1, "no": 0})


# Convert furnishing status into dummy variables
df = pd.get_dummies(
    df,
    columns=["furnishingstatus"],
    drop_first=True,
    dtype=int
)

print("\n================ AFTER PREPROCESSING =================")
print(df.head())



# Dependent variable
y = df["price"]

# Independent variables
X = df.drop("price", axis=1)

print("\nIndependent variables:")
print(X.columns.tolist())

print("\nDependent variable:")
print("price")



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n================ TRAIN-TEST SPLIT =================")
print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))



model = LinearRegression()

model.fit(X_train, y_train)



print("\n================ REGRESSION COEFFICIENTS =================")

print("Intercept:", model.intercept_)

coefficient_table = pd.DataFrame({
    "Variable": X.columns,
    "Coefficient": model.coef_
})

print(coefficient_table.to_string(index=False))


y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)




train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

train_rmse = np.sqrt(
    mean_squared_error(y_train, y_train_pred)
)

test_rmse = np.sqrt(
    mean_squared_error(y_test, y_test_pred)
)

print("\n================ MODEL ACCURACY =================")

print("\nTraining Set:")
print("R²   :", round(train_r2, 4))
print("MAE  :", round(train_mae, 2))
print("RMSE :", round(train_rmse, 2))

print("\nTesting Set:")
print("R²   :", round(test_r2, 4))
print("MAE  :", round(test_mae, 2))
print("RMSE :", round(test_rmse, 2))



X_stats = sm.add_constant(X)

ols_model = sm.OLS(y, X_stats).fit()

print("\n================ OLS REGRESSION SUMMARY =================")

print(ols_model.summary())




results = pd.DataFrame({
    "Variable": ols_model.params.index,
    "Coefficient": ols_model.params.values,
    "P-value": ols_model.pvalues.values
})

results["Significant"] = results["P-value"] < 0.05

print("\n================ VARIABLE SIGNIFICANCE =================")

print(results.to_string(index=False))

print("\nVariables significant at 5% level:")

significant_variables = results[
    (results["P-value"] < 0.05) &
    (results["Variable"] != "const")
]

print(significant_variables["Variable"].tolist())




# Example property
# Modify these values as required

new_property = pd.DataFrame({
    "area": [4000],
    "bedrooms": [4],
    "bathrooms": [2],
    "stories": [2],
    "mainroad": [1],
    "guestroom": [1],
    "basement": [0],
    "hotwaterheating": [0],
    "airconditioning": [1],
    "parking": [2],
    "prefarea": [1],
    "furnishingstatus_semi-furnished": [0],
    "furnishingstatus_unfurnished": [0]
})

# Make sure columns are in the same order as training data
new_property = new_property[X.columns]

predicted_price = model.predict(new_property)

print("\n================ NEW PROPERTY PREDICTION =================")

print("Property details:")
print(new_property)

print("\nPredicted House Price:")
print("₹", round(predicted_price[0], 2))


comparison = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_test_pred
})

print("\n================ ACTUAL VS PREDICTED =================")

print(comparison.head(10))




plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_test_pred,
    alpha=0.7
)

# Perfect prediction line
min_price = min(y_test.min(), y_test_pred.min())
max_price = max(y_test.max(), y_test_pred.max())

plt.plot(
    [min_price, max_price],
    [min_price, max_price],
    linestyle="--"
)

plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs Predicted House Prices")

plt.tight_layout()
plt.show()




residuals = y_test - y_test_pred

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test_pred,
    residuals,
    alpha=0.7
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Predicted House Price")
plt.ylabel("Residual")
plt.title("Residual Plot")

plt.tight_layout()
plt.show()


print("\n===================================================")
print("FINAL MODEL SUMMARY")
print("===================================================")

print("Number of observations :", len(df))
print("Number of predictors   :", X.shape[1])

print("\nTraining R² :", round(train_r2, 4))
print("Testing R²  :", round(test_r2, 4))

print("\nTesting MAE  :", round(test_mae, 2))
print("Testing RMSE :", round(test_rmse, 2))

print("\nThe model explains approximately",
      round(test_r2 * 100, 2),
      "% of the variation in house prices on the test data.")