import os
import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.core.metrics import make_scorer
from sklearn.metrics import root_mean_squared_log_error
from openfe import OpenFE, transform

if __name__ == '__main__':
    os.chdir('..')
    train_data_path: str = 'playground-series-s4e4/train.csv'  # Path to train data csv file
    test_data_path = 'playground-series-s4e4/test.csv' # Path to test data csv file
    submission_save_directory = 'seminararbeit/submissions/autogluon/submission.csv' # directory in which the submission will get saved
    n_jobs: int = 4  # Number of used CPU cores for feature engineering
    use_openfe: bool = True  # Decide to use openfe
    data = TabularDataset(pd.read_csv(train_data_path)).drop(columns='id')  # Load Training Data
    X_test = TabularDataset(pd.read_csv(test_data_path)).drop(columns='id')  # Load Test Data
    if use_openfe:
        X = data
        y = data.pop('Rings')  # Split dataset into features X and target y
        features = OpenFE().fit(X, y, task='regression', n_jobs=n_jobs)  # Select useful features
        X, X_test = transform(X, X_test, features, n_jobs=n_jobs)  # Transform original data
        data = pd.concat([X, y], axis=1)  # Concatenate data to one dataset


    # Initialize Autogluon Tabular Regressor
    model = TabularPredictor('Rings', eval_metric=make_scorer('rmsle', root_mean_squared_log_error,
                                                              greater_is_better=False), problem_type='regression')
    model.fit(data, presets='best_quality')  # Fit regressor to data
    print(model.fit_summary()['leaderboard'].to_string())  # Print training summary

    predictions = model.predict(X_test) # Make predictions on test data
    submission_dataframe = pd.read_csv('playground-series-s4e4/sample_submission.csv') # Load sample submission
    submission_dataframe['Rings'] = predictions # Set autogluon submissions as predictions
    try:
        submission_dataframe.to_csv(submission_save_directory,index=False)  # Save submission csv
    except OSError:
        os.mkdir(submission_save_directory)
        submission_dataframe.to_csv(submission_save_directory,index=False)  # Save submission csv