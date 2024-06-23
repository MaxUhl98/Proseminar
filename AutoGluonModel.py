import sys
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import root_mean_squared_log_error
from openfe import OpenFE, transform
from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.core.metrics import make_scorer
from configuration import cfg
from fastai.metrics import msle


def get_autogluon_scorer():
    return make_scorer(name='rmsle', score_func=root_mean_squared_log_error, optimum=0, greater_is_better=False,
                       needs_pred=True)


def msle_loss(y_true, y_pred):
    y_pred = np.maximum(y_pred, -1 + 1e-6)
    return ((np.log1p(y_pred) - np.log1p(y_true)) / (1 + y_pred),
            (1 - np.log1p(y_pred) + np.log1p(y_true)) / (1 + y_pred) ** 2)


if __name__ == '__main__':
    xgboost_options = {'objective': 'reg:squaredlogerror'}
    hyperparameters = {'GBM': {}, 'XGB': xgboost_options, 'CAT': {'loss_function': 'RMSE'}, 'RF': {},'XT':{},
                       'NN_TORCH': {}, 'FASTAI':{}}
    df = pd.read_csv('playground-series-s4e4/train.csv')
    df = df.drop(columns='id')
    #df = pd.get_dummies(df)
    ag_rmsle_scorer = get_autogluon_scorer()
    model = TabularPredictor(eval_metric=ag_rmsle_scorer, label='Rings', path='models', problem_type='regression',
                             )
    #df = df.loc[df['Rings'] <= 20]
    y = df['Rings']
    X = df.drop(columns='Rings')
    #X[['Sex_F', 'Sex_I', 'Sex_M']] = X[['Sex_F', 'Sex_I', 'Sex_M']].astype(np.float64)
    ofe = OpenFE()
    features = ofe.fit(X, y, task='regression', verbose=False, n_jobs=8)
    X, _ = ofe.transform(X, X.iloc[0:0], features[:cfg.n_features], n_jobs=cfg.n_jobs)

    X = TabularDataset(pd.concat([X, y], axis=1))
    model = model.fit(X, presets=cfg.quality, hyperparameters=hyperparameters)

    # Preprocess the rows identically to the train data
    df_submission = pd.read_csv('playground-series-s4e4/test.csv')

    # Save Submission id's to add them to the submission dataframe later (required by the competition for identification)
    submission_ids = df_submission['id']

    X_submission = df_submission.drop(columns='id')
    X_submission = pd.get_dummies(X_submission)

    # Use feature engineering for folds
    X_sub = transform(X_submission, X_submission, features[:cfg.n_features], n_jobs=cfg.n_jobs)[0]

    # Use the averaged predictions across all folds
    preds = model.predict(X_sub)
    df_submission = pd.DataFrame({'id': submission_ids, 'prediction': preds})

    # Create Submission Dataframe
    df_submission.index = df_submission.id
    df_submission.drop(columns='id', inplace=True)
    df_submission.head()

    # Write Submission to CSV file
    df_submission.to_csv('submission.csv')
