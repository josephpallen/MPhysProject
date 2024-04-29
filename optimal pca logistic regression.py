# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 21:16:26 2024

@author: joeal
"""

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.metrics import make_scorer, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler # Import RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, plot_confusion_matrix, roc_curve, precision_recall_curve, auc, average_precision_score
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
df = pd.read_csv('Classification Training set.csv')

# Assuming 'ML_ID' is the target column for prediction
X = df.drop(['ML_ID', 'DR3Name'], axis=1)
y = df['ML_ID']

# Split the dataset into training and testing sets with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42, stratify=y)

# Scale the input features using RobustScaler instead of StandardScaler
scaler = RobustScaler() # Use RobustScaler
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the hyperparameters for the grid search
param_grid = {
    'penalty': ['l1'],
    'C': [1],
    'solver': ['liblinear'],
    'class_weight':['balanced'],
    'fit_intercept':[True],
    'tol':[0.01]
}

# Initialize the logistic regression model
log_reg = LogisticRegression()

# Define the Weighted F1 Score
weighted_f1_scorer = make_scorer(f1_score, average='weighted')

# Initialize the grid search with the model and parameters
grid_search = GridSearchCV(log_reg, param_grid, cv=4, scoring=weighted_f1_scorer, n_jobs=-1, verbose=2)
# Lists to store validation scores and the corresponding number of PCA components
validation_scores = []
n_components_list = []

# Loop through different numbers of PCA components
for n_components in range(1, 40): # Adjust the range as needed
    # Apply PCA to reduce the dimensionality of the input features
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # Fit the grid search to the PCA-transformed training data
    grid_search.fit(X_train_pca, y_train)

    # Predict on the PCA-transformed test dataset using the best model
    best_model = grid_search.best_estimator_
    predictions = best_model.predict(X_test_pca)

    # Calculate the validation score
    validation_score = best_model.score(X_test_pca, y_test)

    # Store the validation score and the number of PCA components
    validation_scores.append(validation_score)
    n_components_list.append(n_components)

# Find the best number of PCA components
best_n_components = n_components_list[np.argmax(validation_scores)]
best_validation_score = np.max(validation_scores)

# Use the best number of PCA components to apply PCA again, fit the model, and generate the classification report and confusion matrix
pca = PCA(n_components=best_n_components)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

grid_search.fit(X_train_pca, y_train)
best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test_pca)

# Plot Validation Score vs. Number of Components
plt.figure()
plt.plot(n_components_list, validation_scores, marker='o')
plt.xlabel('Number of PCA Components')
plt.ylabel('Validation Score')
plt.title('Validation Score vs. Number of PCA Components')
plt.grid(True)
plt.show()

# Print the best number of PCA components and the corresponding validation score
print(f"Best number of PCA components: {best_n_components} with validation score: {best_validation_score}")

# Print the classification report
print(classification_report(y_test, predictions))

# Plot the confusion matrix
plot_confusion_matrix(best_model, X_test_pca, y_test, cmap=plt.cm.Reds)
plt.title('Confusion Matrix for Best PCA Components')
plt.show()

# Plot the learning curve for the best PCA number
train_sizes, train_scores, test_scores = learning_curve(best_model, X_train_pca, y_train, cv=4)
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
plt.figure()
plt.plot(train_sizes, train_scores_mean, label='Training score')
plt.plot(train_sizes, test_scores_mean, label='Cross-validation score')
plt.xlabel('Training set size')
plt.ylabel('Score')
plt.title('Learning Curve for Best PCA Components')
plt.legend(loc="best")
plt.show()

# Plot multiclass ROC curves
plt.figure()
for i in range(len(np.unique(y_test))):
    fpr, tpr, _ = roc_curve(y_test == i, predictions == i)
    plt.plot(fpr, tpr, label=f'Class {i} (area = {auc(fpr, tpr):.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multiclass ROC Curves')
plt.legend(loc="lower right")
plt.show()

# Plot multiclass precision-recall curves
plt.figure()
for i in range(len(np.unique(y_test))):
    precision, recall, _ = precision_recall_curve(y_test == i, predictions == i)
    plt.plot(recall, precision, label=f'Class {i} (area = {average_precision_score(y_test == i, predictions == i):.2f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Multiclass Precision-Recall Curves')
plt.legend(loc="lower left")
plt.show()

# Plot cumulative explained variance
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance for Best PCA Components')
plt.show()