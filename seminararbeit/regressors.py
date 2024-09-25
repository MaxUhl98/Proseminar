import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_log_error
from xgboost import XGBRegressor
from openfe import OpenFE, transform

if __name__ == '__main__':
    # Model initialization keyword arguments
    model_kwargs = {XGBRegressor: {'objective': 'reg:squaredlogerror'}, LinearRegression: {}}
    n_jobs: int = 4  # Number of CPU Cores used for Feature Engineering
    use_openfe: bool = False  # Decide to use openfe
    model = XGBRegressor  # XGBRegressor to perform Gradient Boosting, LinearRegression to perform Linear Regression
    folder = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)  # Initialize Splitter for Cross Validation

    X = pd.read_csv('playground-series-s4e4/train.csv').drop(columns='id')  # X:= Features, y:= Target
    y = X.pop('Rings')
    if use_openfe:
        # Preprocess Features with openfe
        features = OpenFE().fit(X, y, task='regression', n_jobs=n_jobs) # Select useful candidate features
        X, _ = transform(X, X.iloc[0:0], features, n_jobs=n_jobs)  # Transform dataset using openfe generated features
    X = pd.get_dummies(X)  # One hot encoding
    splits = folder.split(X, y)  # Create Cross Validation Splits
    fold_scores = []

    for train, test in splits:
        # Select train and test data according to CV indices
        X_train, X_val = X.iloc[train], X.iloc[test]
        y_train, y_val = y.iloc[train], y.iloc[test]

        if model == LinearRegression:
            # Log transform target to help the regression out (regression always uses MSE)
            y_train, y_val = np.log1p(y_train), np.log1p(y_val)

        current_model = model(**model_kwargs[model])  # Initialize Model with keyword arguments from line 11
        current_model.fit(X_train, y_train)  # Train model

        # Get predictions
        pred = current_model.predict(X_val) if model == XGBRegressor else np.expm1(current_model.predict(X_val))

        fold_rmsle = root_mean_squared_log_error(y_val, pred)  # Calculate RMSLE
        fold_scores.append(fold_rmsle)

    print(f'Val RMSLE: {np.mean(fold_scores)}')  # Print mean RMSLE across all folds
