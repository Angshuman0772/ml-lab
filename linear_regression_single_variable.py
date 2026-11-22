# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 09:35:29 2026

@author: I3D-MSI
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


data = {
    'YearsExperience': [
        1.1, 1.3, 1.5, 2.0, 2.2, 2.9, 3.0, 3.2, 3.2, 3.7,
        3.9, 4.0, 4.0, 4.1, 4.5, 4.9, 5.1, 5.3, 5.9, 6.0,
        6.8, 7.1, 7.9, 8.2, 8.7, 9.0, 9.5, 9.6, 10.3, 10.5
    ],
    
    'Salary': [
        39343, 46205, 37731, 43525, 39891, 56642, 60150, 54445,
        64445, 57189, 63218, 55794, 56957, 57081, 61111, 67938,
        66029, 83088, 81363, 93940, 91738, 98273, 101302, 113812,
        109431, 105582, 116969, 112635, 122391, 121872
    ]
}

df = pd.DataFrame(data)

X = df[['YearsExperience']]
y = df['Salary']



model = LinearRegression()

model.fit(X, y)



intercept = model.intercept_
slope = model.coef_[0]

print("\nIntercept:", intercept)
print("Slope:", slope)

# Regression equation
print("\nRegression Equation:")
print(f"Salary = {intercept:.2f} + {slope:.2f} × YearsExperience")



y_pred = model.predict(X)

df['PredictedSalary'] = y_pred

print("\nActual vs Predicted Salary:")
print(df)



r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))

print("\nR² Score:", r2)
print("RMSE:", rmse)



years = [[6.5]]

predicted_salary = model.predict(years)

print("\nPredicted salary for 6.5 years experience:")
print(f"₹{predicted_salary[0]:,.2f}")



plt.scatter(X, y, label='Actual Data')

plt.plot(X, y_pred, label='Regression Line')

plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Linear Regression: Experience vs Salary')

plt.legend()
plt.show()