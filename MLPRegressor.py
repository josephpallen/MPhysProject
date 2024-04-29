# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 14:43:02 2024

@author: joeal
"""

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

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

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the StandardScaler on the training data
scaler.fit(X_train)

# Transform the training and testing data
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the model
mlp = MLPRegressor()

# Define the parameter grid for GridSearchCV
param_grid = {
    'hidden_layer_sizes': [(23,)],
    'activation': ['tanh'],
    'solver': ['adam'],
    'alpha': [0.01],
    'learning_rate': ['invscaling'],
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)

# Fit GridSearchCV
grid_search.fit(X_train_scaled, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")

# Fit the model with the best parameters
best_model = MLPRegressor(**best_params)
best_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = best_model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Plot the scatter plot for the current model with a dashed line of y=x
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Log of Actual Effective Temperature (K)')
plt.ylabel('Log of Predicted Temperature (K)')
plt.show()

# Calculate permutation importance
result = permutation_importance(best_model, X_test_scaled, y_test, n_repeats=10, random_state=42, n_jobs=-1)

# Get the importance of each feature
importance = result.importances_mean

# Get the indices of the top 5 features
top_5_indices = importance.argsort()[-5:][::-1]

# Print the names of the top 5 features
print("Top 5 most important features:")
for index in top_5_indices:
    print(f"{X.columns[index]}: {importance[index]}")