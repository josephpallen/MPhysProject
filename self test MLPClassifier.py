# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 03:26:18 2024

@author: joeal
"""

import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, roc_auc_score, average_precision_score, plot_confusion_matrix, make_scorer, f1_score
from sklearn.preprocessing import RobustScaler
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

# Split the dataset into training and testing sets with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, stratify=y, random_state=42)

# Scale the input features
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the hyperparameters for the grid search
param_grid = {
    'hidden_layer_sizes': [(42,)],
    'activation': ['tanh'],
    'solver': ['lbfgs'],
    'alpha': [0.01],
    'max_iter': [100000],
    'learning_rate': ['constant']
}

# Initialize the MLPClassifier
mlp_clf = MLPClassifier()

# Define the Weighted F1 Score
weighted_f1_scorer = make_scorer(f1_score, average='weighted')

# Initialize the grid search with the model and parameters
grid_search = GridSearchCV(mlp_clf, param_grid, cv=4, scoring=weighted_f1_scorer, n_jobs=-1, verbose=1)

# Fit the grid search to the scaled training data
grid_search.fit(X_train_scaled, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print("Best parameters:", best_params)

# Predict on the scaled test dataset using the best model
best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test_scaled)

# Print the classification report with zero_division parameter
print(classification_report(y_test, predictions))

# Plot the confusion matrix
plot_confusion_matrix(best_model, X_test_scaled, y_test, cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# Multiclass ROC curves
plt.figure()
for i in range(4): # Assuming there are 4 classes (0, 1, 2, 3)
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
for i in range(4): # Assuming there are 4 classes (0, 1, 2, 3)
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
train_sizes, train_scores, test_scores = learning_curve(best_model, X_train_scaled, y_train, cv=4)
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

# Calculate permutation importance
result = permutation_importance(best_model, X_test_scaled, y_test, n_repeats=10, random_state=42, n_jobs=-1)

# Get the importance scores
importance = result.importances_mean

# Get the indices of the top 5 features
top_5_indices = importance.argsort()[-5:][::-1]

# Get the feature names
feature_names = X.columns

# Print the top 5 features
print("Top 5 features:")
for index in top_5_indices:
    print(f"{feature_names[index]}: {importance[index]}")

