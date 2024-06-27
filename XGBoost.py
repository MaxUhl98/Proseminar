import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import StratifiedKFold
import numpy as np
import xgboost as xgb
from sklearn.feature_selection import SequentialFeatureSelector, RFE, RFECV, SelectFromModel, VarianceThreshold
from ucimlrepo import fetch_ucirepo
from utils.helpers import load_ofe
from sklearn.metrics import root_mean_squared_log_error, make_scorer
from openfe import transform
from typing import *
import pickle
from utils.helpers import preprocess_features


def gradient(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
    '''Compute the gradient squared log error.'''
    y = dtrain.get_label()
    return (np.log1p(predt) - np.log1p(y)) / (predt + 1)


def hessian(predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
    '''Compute the hessian for squared log error.'''
    y = dtrain.get_label()
    return ((-np.log1p(predt) + np.log1p(y) + 1) /
            np.power(predt + 1, 2))


def squared_log(predt: np.ndarray,
                dtrain: xgb.DMatrix) -> tuple[np.ndarray, np.ndarray]:
    '''Squared Log Error objective. A simplified version for RMSLE used as
    objective function.

    :math:`\frac{1}{2}[log(pred + 1) - log(label + 1)]^2`

    '''
    predt[predt < -1] = -1 + 1e-6
    grad = gradient(predt, dtrain)
    hess = hessian(predt, dtrain)
    return grad, hess


def rmsle(predt: np.ndarray, dtrain: xgb.DMatrix) -> tuple[str, float]:
    ''' Root mean squared log error metric.

    :math:`\sqrt{\frac{1}{N}[log(pred + 1) - log(label + 1)]^2}`
    '''
    y = dtrain.get_label()
    predt[predt < -1] = -1 + 1e-6
    elements = np.power(np.log1p(y) - np.log1p(predt), 2)
    return 'PyRMSLE', float(np.sqrt(np.sum(elements) / len(y)))


def create_submission(used_columns: List[str]):
    df_submission = pd.read_csv('playground-series-s4e4/test.csv')
    # Save Submission id's to add them to the submission dataframe later (required by the competition for identification)
    submission_ids = df_submission['id']

    ofe, features = load_ofe()

    X_submission = df_submission.drop(columns='id')
    X_sub = transform(X_submission, X_submission.iloc[0:0], features, n_jobs=XGBConfig.n_jobs)[0]
    X_sub = pd.get_dummies(X_sub)
    X_sub = X_sub[used_columns]

    # Use the averaged predictions across all folds
    preds = sum([_model.predict(X_sub.values) for _model in models]) / XGBConfig.num_folds
    df_submission = pd.DataFrame({'id': submission_ids, 'prediction': preds})
    df_submission.index = df_submission.id
    df_submission.drop(columns='id', inplace=True)
    df_submission.to_csv('submissions/xgb/submission.csv')


class XGBConfig:
    num_folds: int = 10
    n_jobs: int = 30
    method: str = 'tuned_params'
    num_engineered_features: int = 10 ** 4
    load_feature_selector: bool = False


if __name__ == '__main__':
    X, y, X_additional_train, y_additional_train = preprocess_features()
    print('Dummied Data')

    folder = StratifiedKFold(n_splits=XGBConfig.num_folds, shuffle=True)
    folds = folder.split(X, y)

    #params = {'reg_lambda': 0.2508574300929533,
    #          'reg_alpha': 0.13159161365843544,
    #          'subsample': 0.5986422285944151,
    #          'colsample_bytree': 0.4552369359133244,
    #          'sampling_method': 'gradient_based',
    #          'max_depth': 9,
    #          'min_child_weight': 2,
    #          'learning_rate': 0.005549096292032568,
    #          'gamma': 1.7388024670776588e-05,
    #          'n_estimators':10**4,
    #          'early_stopping_rounds':100
    #          }

    models = [XGBRegressor(objective='reg:squaredlogerror') for _ in range(XGBConfig.num_folds)]
    print('Created Models')
    X_additional_train = X_additional_train[list(X.columns)]
    fold_data = {'Val RMSLE': []}
    for num, fold_idx in enumerate(folds):
        train_idx, val_idx = fold_idx

        X_train = pd.concat([X.iloc[train_idx], X_additional_train], axis=0)
        y_train = pd.concat([y.iloc[train_idx], y_additional_train], axis=0)

        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]

        models[num].fit(X_train, y_train, eval_set=[(X_val, y_val)])
        fold_data['Val RMSLE'] += [root_mean_squared_log_error(y_val, models[num].predict(X_val.values))]
        print(f'Fold {num + 1} Val RMSLE: {fold_data["Val RMSLE"][-1]}')
    train_history = pd.DataFrame(fold_data)
    print(f'Average Val RMSLE: {train_history["Val RMSLE"].mean()}')

    for num, model in enumerate(models):
        model.save_model(f'models/xgb/{XGBConfig.method}_fold_{num + 1}_xgb.json')
    create_submission(used_columns=list(X.columns))
