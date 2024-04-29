# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 16:56:42 2024

@author: joeal
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler

# Load the datasets
train_df = pd.read_csv('Classification Training Set.csv')
test_df = pd.read_csv('Classification Testing Set.csv')

# Drop 'DR3Name' and 'ML_ID' from both datasets
train_df.drop(['DR3Name'], axis=1, inplace=True)
test_df.drop(['DR3Name', 'ML_ID'], axis=1, inplace=True)

# Assuming 'ML_ID' is the target column for prediction
X_train = train_df.drop('ML_ID', axis=1)
y_train = train_df['ML_ID']

# Initialize the RobustScaler
scaler = RobustScaler()

# Fit the scaler on the training data and transform both the training and test data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(test_df)

# Train the Random Forest Classifier model and make predictions 10 times
for i in range(10):
    rf_clf = RandomForestClassifier()
    rf_clf.fit(X_train_scaled, y_train) # Ensure the model is trained

    # Predict on the test dataset
    predictions = rf_clf.predict(X_test_scaled)

    # Append the predictions to a new column in the test_df DataFrame
    test_df[f'Predicted_Class_{i+1}'] = predictions

# Calculate the number of each class identified for the last prediction
class_counts = pd.Series(predictions).value_counts()

# Print the number of classes identified
print(f"Number of Classes Identified in the Last Prediction: {class_counts}")

# Save the modified test_df DataFrame to a file
test_df.to_csv('random forest candidates.csv', index=False)
