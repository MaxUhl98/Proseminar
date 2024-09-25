import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.core.metrics import make_scorer
from sklearn.metrics import root_mean_squared_log_error
from openfe import OpenFE, transform

if __name__ == '__main__':
    data_path: str = 'playground-series-s4e4/train.csv'  # Path to data csv file
    n_jobs: int = 4  # Number of used CPU cores for feature engineering
    use_openfe: bool = True  # Decide to use openfe
    data = TabularDataset(pd.read_csv(data_path)).drop(columns='id')  # Load Data
    if use_openfe:
        X = data
        y = data.pop('Rings')  # Split dataset into features X and target y
        features = OpenFE().fit(X, y, task='regression', n_jobs=n_jobs)  # Select useful features
        X, _ = transform(X, X.iloc[0:0], features, n_jobs=n_jobs)  # Transform original data
        data = pd.concat([X, y], axis=1)  # Concatenate data to one dataset

    # Initialize Autogluon Tabular Regressor
    model = TabularPredictor('Rings', eval_metric=make_scorer('rmsle', root_mean_squared_log_error,
                                                              greater_is_better=False), problem_type='regression')
    model.fit(data, presets='best_quality')  # Fit regressor to data
    print(model.fit_summary()['leaderboard'].to_string())  # Print training summary
