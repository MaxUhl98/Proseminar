import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_log_error
from xgboost import XGBRegressor
from openfe import OpenFE, transform
from utils.helpers import create_submission

if __name__ == '__main__':
    os.chdir('..')
    # Model initialization keyword arguments
    model_kwargs = {XGBRegressor: {'objective': 'reg:squaredlogerror'}, LinearRegression: {}}
    n_jobs: int = 4  # Number of CPU Cores used for Feature Engineering
    use_openfe: bool = False  # Decide to use openfe
    model = LinearRegression # XGBRegressor to perform Gradient Boosting, LinearRegression to perform Linear Regression
    num_folds:int = 10
    folder = StratifiedKFold(n_splits=num_folds, random_state=1, shuffle=True)  # Initialize Splitter for Cross Validation

    X = pd.read_csv('playground-series-s4e4/train.csv').drop(columns='id')  # X:= Features, y:= Target
    y = X.pop('Rings')
    X_test = pd.read_csv('playground-series-s4e4/test.csv').drop(columns='id')  # Load Test Data
    if use_openfe:
        # Preprocess Features with openfe
        features = OpenFE().fit(X, y, task='regression', n_jobs=n_jobs) # Select useful candidate features
        X, X_test = transform(X, X_test, features, n_jobs=n_jobs)  # Transform dataset using openfe generated features
    X = pd.get_dummies(X)  # One hot encoding
    X_test = pd.get_dummies(X_test)
    splits = folder.split(X, y)  # Create Cross Validation Splits
    fold_scores = []
    trained_models = []

    for train, test in splits:
        # Select train and test data according to CV indices
        X_train, X_val = X.iloc[train], X.iloc[test]
        y_train, y_val = y.iloc[train], y.iloc[test]

        if model == LinearRegression:
            # Log transform target to help the regression out (regression always uses MSE)
            y_train, y_val = np.log1p(y_train), y_val

        current_model = model(**model_kwargs[model])  # Initialize Model with keyword arguments from line 11
        current_model.fit(X_train, y_train)  # Train model
        trained_models.append(current_model) # Add current model to trained model list

        # Get predictions
        pred = current_model.predict(X_val) if model == XGBRegressor else np.expm1(current_model.predict(X_val))

        fold_rmsle = root_mean_squared_log_error(y_val, pred)  # Calculate RMSLE
        fold_scores.append(fold_rmsle)

    print(f'Val RMSLE: {np.mean(fold_scores)}')  # Print mean RMSLE across all folds

    if model == LinearRegression:
        model_predictions = [np.expm1(_model.predict(X_test.values)) for _model in trained_models]
    else:
        model_predictions = [_model.predict(X_test.values) for _model in trained_models]

    model_predictions = sum(model_predictions) / num_folds
    submission_dataframe = pd.read_csv('playground-series-s4e4/sample_submission.csv')  # Load sample submission
    submission_dataframe['Rings'] = model_predictions  # Set autogluon submissions as predictions
    submission_save_directory = f'seminararbeit/submissions/{model.__name__}/submission.csv'
    try:
        submission_dataframe.to_csv(submission_save_directory,index=False)  # Save submission csv
    except OSError:
        os.mkdir(submission_save_directory)
        submission_dataframe.to_csv(submission_save_directory,index=False)  # Save submission csv

