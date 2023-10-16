""" Completing Microsoft Learn"s Wine Classification Chllenge.
    The goal is to achieve a recall metric of over 0.95.

Data originally from:
    Forina, M. et al.
    PARVUS - An Extendible Package for Data Exploration,
             Classification and Correlation.
    Institute of Pharmaceutical and Food Analysis and Technologies,
    Via Brigata Salerno, 16147 Genoa, Italy.

"""

# Standard library imports
import os

# External imports
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Constants
REGULARIZATION_RATE = 0.01

# load data
data = pd.read_csv(os.path.join("data", "wine.csv"))

# Extract training features
features = data.columns.values.tolist()

# Separate WineVariety data from other characteristics
x, y = data[features].values, data["WineVariety"].values

# Split the data into test and train
# (setting random_state to see consistent inputs
#   whilst trying different training techniques)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.30, random_state=0)

# Train LogisticRegression model, but scale numeric features
pipeline = Pipeline(steps=[
    ('standardscaler', StandardScaler()),
    ('logregressor', LogisticRegression(
        C=1/REGULARIZATION_RATE, solver="liblinear"))])
model = pipeline.fit(x_train, y_train)

# Predict using our verification (test) data
predictions = model.predict(x_test)

# Compare our predictions to the correct values
print(f"Prediction: \n{predictions}")
print(f"Correct: \n{y_test}")

# Print accuracy score, where 1 is 100% accurate and 0 is 0% accurate
print(f"\nAccuracy score: {round(accuracy_score(y_test, predictions), 3)}")

# Print confusion matrix for visual representation of
# false positives, false negatives, true positive, and true negatives
calculated_confusion_matrix = confusion_matrix(y_test, predictions)
print(calculated_confusion_matrix)
