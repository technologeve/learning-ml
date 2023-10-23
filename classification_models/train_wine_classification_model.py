""" Completing Microsoft Learn's Wine Classification Challenge.
    The goal is to achieve a recall metric of over 0.95.

Data originally from:
    Forina, M. et al.
    PARVUS - An Extendible Package for Data Exploration,
             Classification and Correlation.
    Institute of Pharmaceutical and Food Analysis and Technologies,
    Via Brigata Salerno, 16147 Genoa, Italy.

Training two classification models on sample data.

"""

# Standard library imports
import os

# External imports
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Constants
REGULARIZATION_RATE = 0.01


def load_data_and_extract_features(filepath):
    """ Load data from a csv file and split into
        testing and training data (30%:70%)

    Keyword arguments:
    filepath:   A string containing the filepath, filename,
                and filetype of the file containing the data of interest

    Outputs:
    x_train, x_test, y_train, y_test
    """

    # load data
    data = pd.read_csv(filepath)

    # Extract training features
    features = data.columns.values.tolist()

    # Separate WineVariety data from other characteristics
    x, y = np.delete(
        data[features].values, -1, axis=1), data["WineVariety"].values

    # Split the data into test and train
    # (setting random_state to see consistent inputs
    #   whilst trying different training techniques)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.30, random_state=0)

    return x_train, x_test, y_train, y_test


def train_model(model_class, x_train, y_train):
    """ Preprocess by scaling all x_train data as numeric features.
        Then train already declared model, on x_train and y_train model data.

    Keyword arguments:
    model_class: Declared classification model of choice,
                 including any arguments.
                 Eg. RandomForestClassifier(n_estimators=100)
    x_train:     The input training data
    y_train:     The training labels

    Outputs:
    model:       Classification model of choice fit to x and y training data.

    """

    # Train classification model, but scale numeric features
    pipeline = Pipeline(steps=[
        ('standardscaler', StandardScaler()),
        ('logregressor', model_class)])

    # Fit model to x and y training data
    model = pipeline.fit(x_train, y_train)

    # Return fitted model
    return model


def evaluate_and_print_model(y_test, predictions):
    """ Compare model's predictions with correct y_test data. """

    # Compare our predictions to the correct values
    print(f"Prediction: \n{predictions}")
    print(f"Correct: \n{y_test}")

    # Print accuracy score, where 1 is 100% accurate and 0 is 0% accurate
    accuracy = accuracy_score(y_test, predictions)
    print(f"\nAccuracy score: {round(accuracy, 3)}")

    # Print confusion matrix for visual representation of
    # false positives, false negatives, true positive, and true negatives
    calculated_confusion_matrix = confusion_matrix(y_test, predictions)
    print(calculated_confusion_matrix)

    return accuracy, calculated_confusion_matrix


def save_model(model, filename):
    """ Save trained model as a pickle file

    Keyword arguments:
    model:      Trained model instance to be saved
    filename:   File name you would like to save the model under

    """
    joblib.dump(model, filename)


# Load in wine classification data
x_train, x_test, y_train, y_test = load_data_and_extract_features(
    os.path.join("data", "wine.csv"))

# Train LogisticRegression model, predict using test data, and evaluate model
model = train_model(LogisticRegression(
    C=1/REGULARIZATION_RATE, solver="liblinear"), x_train, y_train)
predictions = model.predict(x_test)
evaluate_and_print_model(y_test, predictions)
save_model(model, 'wine_logistic_regression_model.pkl')

# Train RandomForestClassifier model, predict using test data,
# and evaluate model
model = train_model(RandomForestClassifier(n_estimators=100), x_train, y_train)
predictions = model.predict(x_test)
evaluate_and_print_model(y_test, predictions)
save_model(model, 'wine_random_forest_model.pkl')
