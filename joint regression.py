# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 15:45:09 2024

@author: joeal
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import time # Import the time module

# Load the dataset
df = pd.read_csv('Reduced Regression Training Set.csv')

# Replace infinity values with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Remove rows with NaN values
df.dropna(inplace=True)

# Split the dataset into features and target variable
X = df.drop(['Teff'], axis=1)
y = df['Teff']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

# Initialize the RobustScaler
scaler = RobustScaler()

# Fit the RobustScaler on the training data
scaler.fit(X_train)

# Transform the training and testing data
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the models
linear_regression = LinearRegression()
random_forest = RandomForestRegressor()
mlp = MLPRegressor(activation='tanh', alpha=0.01, hidden_layer_sizes=(23,), learning_rate='invscaling', max_iter=100000, solver='adam', random_state=42)

# Measure and print the time taken for each model to converge
print("Fitting models...")
start_time = time.time()
linear_regression.fit(X_train_scaled, y_train)
end_time = time.time()
print(f"Linear Regression took {end_time - start_time:.2f} seconds to converge.")

start_time = time.time()
random_forest.fit(X_train_scaled, y_train)
end_time = time.time()
print(f"Random Forest Regressor took {end_time - start_time:.2f} seconds to converge.")

start_time = time.time()
mlp.fit(X_train_scaled, y_train)
end_time = time.time()
print(f"MLPRegressor took {end_time - start_time:.2f} seconds to converge.")

# Make predictions
y_pred_linear = linear_regression.predict(X_test_scaled)
y_pred_random_forest = random_forest.predict(X_test_scaled)
y_pred_mlp = mlp.predict(X_test_scaled)

# Evaluate the models
mse_linear = mean_squared_error(y_test, y_pred_linear)
mse_random_forest = mean_squared_error(y_test, y_pred_random_forest)
mse_mlp = mean_squared_error(y_test, y_pred_mlp)

print(f"Linear Regression MSE: {mse_linear}")
print(f"Random Forest Regressor MSE: {mse_random_forest}")
print(f"MLPRegressor MSE: {mse_mlp}")

# Plotting predicted vs actual values on the same plot
plt.figure(figsize=(10, 6))

# Actual vs Predicted for Linear Regression
plt.scatter(y_test, y_pred_linear, alpha=0.5, label='Linear Regression', color='blue')

# Actual vs Predicted for Random Forest Regressor
plt.scatter(y_test, y_pred_random_forest, alpha=0.5, label='Random Forest', color='red')

# Actual vs Predicted for MLPRegressor
plt.scatter(y_test, y_pred_mlp, alpha=0.5, label='MLPRegressor', color='green')

# Plotting a line of y=x for reference
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)

plt.xlabel('Log of Actual Effective Temperature (K)')
plt.ylabel('Log of Predicted Temperature (K)')
plt.legend()

plt.figure()
# Calculate residuals
residuals_linear = y_test - y_pred_linear
residuals_random_forest = y_test - y_pred_random_forest
residuals_mlp = y_test - y_pred_mlp

# Plotting residuals
plt.figure(figsize=(15, 10))

# Residuals for Linear Regression
plt.subplot(2, 2, 1)
plt.scatter(y_test, residuals_linear, alpha=0.5, color='blue')
plt.axhline(y=0, color='k', linestyle='--')
plt.xlabel('Log of Actual Effective Temperature (K)')
plt.ylabel('Residuals (Linear Regression)')
plt.title('Residuals for Linear Regression')

# Residuals for Random Forest Regressor
plt.subplot(2, 2, 2)
plt.scatter(y_test, residuals_random_forest, alpha=0.5, color='red')
plt.axhline(y=0, color='k', linestyle='--')
plt.xlabel('Log of Actual Effective Temperature (K)')
plt.ylabel('Residuals (Random Forest)')
plt.title('Residuals for Random Forest Regressor')

# Residuals for MLPRegressor
plt.subplot(2, 2, 3)
plt.scatter(y_test, residuals_mlp, alpha=0.5, color='green')
plt.axhline(y=0, color='k', linestyle='--')
plt.xlabel('Log of Actual Effective Temperature (K)')
plt.ylabel('Residuals (MLPRegressor)')
plt.title('Residuals for MLPRegressor')

plt.tight_layout()
plt.show()