# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 14:31:21 2024

@author: joeal
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler # Import RobustScaler

# Load the datasets
train_df = pd.read_csv('Classification Training Set.csv')
test_df = pd.read_csv('Classification Testing Set.csv')

# Drop 'DR3Name' and 'ML_ID' from both datasets
train_df.drop(['DR3Name'], axis=1, inplace=True)
test_df.drop(['DR3Name', 'ML_ID'], axis=1, inplace=True)

# Assuming 'ML_ID' is the target column for prediction
X_train = train_df.drop('ML_ID', axis=1)
y_train = train_df['ML_ID']

# Initialize the LogisticRegression model with specific hyperparameters
log_reg = LogisticRegression(max_iter=10000)

# Initialize the RobustScaler
scaler = RobustScaler()

# Fit the scaler on the training data and transform both training and testing data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(test_df)

# Fit the model on the scaled training data
log_reg.fit(X_train_scaled, y_train)

# Predict on the scaled testing dataset
predictions = log_reg.predict(X_test_scaled)

# Assign the predictions to the 'ML_ID' column of the test_df DataFrame
test_df['ML_ID'] = predictions

# Calculate the number of each class identified
class_counts = pd.Series(predictions).value_counts()

# Print the number of classes identified
print(f"Number of Classes Identified: {class_counts}")

# Save the modified test_df DataFrame to a CSV file
test_df.to_csv('test_df_with_predictions.csv', index=False)

print("Predictions have been appended to the 'ML_ID' column in the test dataset. The dataset has been saved to 'test_df_with_predictions.csv'.")