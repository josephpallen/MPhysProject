# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 00:34:12 2024

@author: joeal
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import RFECV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

plt.rcParams["font.family"]="DeJavu Serif"
plt.rcParams["font.serif"]="Times new Roman"

# Load the dataset
df = pd.read_csv(R.csv')

# Replace infinity values with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Remove rows with NaN values
df.dropna(inplace=True)

# Split the dataset into features and target variable
X = df.drop(['Teff'],axis=1)
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

# Initialize the RandomForestRegressor
rf = RandomForestRegressor()

# Perform feature selection with RFECV on the scaled data
selector = RFECV(estimator=rf, step=1, cv=5, scoring='neg_mean_squared_error')
selector = selector.fit(X_train_scaled, y_train)

# Get the selected features
selected_features = X_train.columns[selector.support_]

# Train the RandomForestRegressor on the selected features
rf.fit(X_train_scaled[:, selector.support_], y_train)

# Predict on the testing set
y_pred = rf.predict(X_test_scaled[:, selector.support_])

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Print the top 5 most important features
print("Top 5 Features:")
feature_ranking = sorted(zip(X.columns, selector.ranking_), key=lambda x: x[1])
for feature, rank in feature_ranking[:5]:
    print(f"Feature '{feature}' is ranked {rank}")

# Plot the scatter plot for the current model with a dashed line of y=x
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='red')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Log of Actual Effective Temperature (K)')
plt.ylabel('Log of Predicted Temperature (K)')
plt.show()

# Plot the number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score")
plt.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_, color='red')
plt.show()