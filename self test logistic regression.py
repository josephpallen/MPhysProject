# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 18:18:00 2024

@author: joeal
"""

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.metrics import make_scorer, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, roc_auc_score, average_precision_score, plot_confusion_matrix
from sklearn.preprocessing import RobustScaler # Import RobustScaler
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"]="DeJavu Serif"
plt.rcParams["font.serif"]="Times new Roman"

# Load the dataset
df = pd.read_csv('Classification Training Set.csv')

# Assuming 'ML_ID' is the target column for prediction
X = df.drop(['ML_ID', 'DR3Name'], axis=1)
y = df['ML_ID']

# Apply RobustScaler to the features
scaler = RobustScaler() # Initialize RobustScaler
X_scaled = scaler.fit_transform(X) # Fit and transform the features

# Split the dataset into training and testing sets with stratification
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.8, random_state=42, stratify=y)

# Define the hyperparameters for the grid search
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100, 1000], # Increased the range of C values
    'penalty': ['l1', 'l2'], # Added 'l1' regularization to explore
    'fit_intercept': [True, False], # Added False to explore
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'], # Added more solvers to explore
    'tol': [1e-4, 1e-3, 1e-2], # Increased the range of tolerance levels
    'class_weight': ['balanced', None], # Added None to explore
}

# Initialize the logistic regression model
log_reg = LogisticRegression()

# Define the Weighted F1 Score
weighted_f1_scorer = make_scorer(f1_score, average='weighted')

# Initialize the grid search with the model and parameters
grid_search = GridSearchCV(log_reg, param_grid, cv=4, scoring=weighted_f1_scorer, n_jobs=-1, verbose=2)

# Fit the grid search to the scaled training data
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print("Best parameters:", best_params)

# Predict on the scaled test dataset using the best model
best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test)

# Print the classification report with zero_division parameter
print(classification_report(y_test, predictions))

# Plot the confusion matrix
plot_confusion_matrix(best_model, X_test, y_test, cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# Multiclass ROC curves
plt.figure()
for i in range(4):
    fpr, tpr, _ = roc_curve(y_test == i, predictions == i)
    plt.plot(fpr, tpr, label=f'Class {i} (area = {auc(fpr, tpr):.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Multiclass precision-recall curves
plt.figure()
for i in range(4):
    precision, recall, _ = precision_recall_curve(y_test == i, predictions == i)
    plt.plot(recall, precision, label=f'Class {i} (area = {average_precision_score(y_test == i, predictions == i):.2f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.show()

# Multiclass learning curves
train_sizes, train_scores, test_scores = learning_curve(best_model, X_train, y_train, cv=4)
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
plt.figure()
plt.plot(train_sizes, train_scores_mean, label='Training score')
plt.plot(train_sizes, test_scores_mean, label='Cross-validation score')
plt.xlabel('Training set size')
plt.ylabel('Score')
plt.title('Learning Curves')
plt.legend(loc="best")
plt.show()

# RFE for feature selection
rfe = RFE(estimator=best_model, n_features_to_select=5, step=1)
rfe = rfe.fit(X_train, y_train)
print("Selected features:", X.columns[rfe.support_])