""" Completing Microsoft Learn"s Wine Classification Chllenge.
    Predicting classes of two previously unseen samples.
"""

# Standard library imports
import os

# External imports
import joblib
import numpy as np

# Previously unseen data, to which we will apply our classification model
unseen_data_1 = np.array([[13.72, 1.43, 2.5, 16.7, 108, 3.4, 3.67,
                           0.19, 2.04, 6.8, 0.89, 2.87, 1285]])
unseen_data_2 = np.array([[12.37, 0.94, 1.36, 10.6, 88, 1.98, 0.57,
                           0.28, 0.42, 1.95, 1.05, 1.82, 520]])

print("Logistic regression model predictions:")

logistic_regression_model = joblib.load(os.path.join(
    "classification_models",
    "trained_models", "wine_logistic_regression_model.pkl"))

print(f"First sample: {logistic_regression_model.predict(unseen_data_1)}")
print(f"Second sample: {logistic_regression_model.predict(unseen_data_2)}")

print("\nRandom forest classifier model predictions:")

random_forest_classifier_model = joblib.load(os.path.join(
    "classification_models",
    "trained_models", "wine_random_forest_model.pkl"))

print(f"First sample: {random_forest_classifier_model.predict(unseen_data_1)}")
print(f"Second sample: {random_forest_classifier_model.predict(unseen_data_2)}")
