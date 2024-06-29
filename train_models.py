import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import StratifiedKFold
import numpy as np
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor, StackingRegressor, VotingRegressor
from utils.helpers import load_ofe
from sklearn.metrics import root_mean_squared_log_error
from openfe import transform
from typing import *
from utils.helpers import preprocess_features, create_submission
import wandb
from typing import *
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression, RidgeCV, Ridge
from catboost import CatBoostRegressor
from configuration import Configuration
import pickle


class TrainConfig:
    method: str = 'vanilla'
    num_engineered_features: int = 10 ** 4
    use_additional_data: bool = True
    select_features: bool = True
    reduce_features: bool = False
    model_class = XGBRegressor
    clipping_range = (1, 25)


def get_model_init_kwargs(cfg: TrainConfig) -> Dict[str, Any]:
    return Configuration.model_init_kwarg_mapping[cfg.model_class]


def need_log_transform(cfg: TrainConfig) -> bool:
    if cfg.model_class in Configuration.log_transform_models:
        return True
    return False


def need_dummies(cfg: TrainConfig) -> bool:
    return True if cfg.model_class in Configuration.dummy_models else False


def need_cv(cfg: TrainConfig) -> bool:
    return False if cfg.model_class in Configuration.automatic_cv_models else True


def load_data(cfg: TrainConfig, dummy_flag: bool) -> tuple[pd.DataFrame, pd.DataFrame, Any, Any, pd.DataFrame]:
    base_dir = Configuration.dummied_data_dir if dummy_flag else Configuration.undummied_data_dir
    if cfg.reduce_features:
        X_path = 'X_reduced'
        y_path = 'y_reduced'
        sub_path = 'X_sub_reduced'
    else:
        X_path = 'X'
        y_path = 'y'
        sub_path = 'X_sub'
    if cfg.use_additional_data:
        X_extra_path = X_path + '_extra'
        y_extra_path = y_path + '_extra'
        X, y = pd.read_feather(f'{base_dir}/{X_path}.feather'), pd.read_feather(f'{base_dir}/{y_path}.feather')
        X_extra, y_extra = pd.read_feather(f'{base_dir}/{X_extra_path}.feather'), pd.read_feather(
            f'{base_dir}/{y_extra_path}.feather')
        sub_path += '_extra'
        return X, y, X_extra, y_extra, pd.read_feather(f'{base_dir}/{sub_path}.feather')
    else:
        X_path += '_raw'
        y_path += '_raw'
        sub_path += '_raw'
        X, y = pd.read_feather(f'{base_dir}/{X_path}.feather'), pd.read_feather(f'{base_dir}/{y_path}.feather')
        return X, y, None, None, pd.read_feather(f'{base_dir}/{sub_path}.feather')


def train(cfg: TrainConfig):
    oof_predictions = []
    log_flag = need_log_transform(cfg)
    cv_flag = need_cv(cfg)
    dummy_flag = need_dummies(cfg)

    X, y, X_additional_train, y_additional_train, X_sub = load_data(cfg=cfg, dummy_flag=dummy_flag)
    assert X.shape[1] == X_sub.shape[1]-1, AssertionError(X.shape, X_sub.shape-1)

    if cfg.select_features:
        try:
            X_sub = X_sub[Configuration.selected_features + ['id']]
            X = X[Configuration.selected_features]
            X_additional_train = X_additional_train[Configuration.selected_features]
        except Exception:
            X = X[Configuration.selected_features]
            X_sub = X_sub[Configuration.selected_features + ['id']]
            pass

    folds = Configuration.folder.split(X, y)

    if cfg.use_additional_data:
        X_additional_train = X_additional_train[list(X.columns)]
    fold_data = {'Val RMSLE': []}

    if dummy_flag:
        X = pd.get_dummies(X)
        X_additional_train = pd.get_dummies(X_additional_train)
    if log_flag:
        y = np.log1p(y)
        y_additional_train = np.log1p(y_additional_train)

    if cv_flag:
        models = [cfg.model_class(**Configuration.model_init_kwarg_mapping[cfg.model_class]) for _ in
                  range(Configuration.num_folds)]
        print('Created Models')
        for num, fold_idx in enumerate(folds):
            train_idx, val_idx = fold_idx
            if cfg.use_additional_data:
                X_train = pd.concat([X.iloc[train_idx], X_additional_train], axis=0)
                y_train = pd.concat([y.iloc[train_idx], y_additional_train], axis=0)
            else:
                X_train = X.iloc[train_idx]
                y_train = y.iloc[train_idx]

            X_val = X.iloc[val_idx]
            y_val = y.iloc[val_idx]

            models[num].fit(X_train, y_train, **Configuration.get_default_fit_kwargs(X_train, cfg.model_class))
            predictions = models[num].predict(X_val.values)
            if log_flag: predictions = np.expm1(predictions)
            predictions = np.clip(predictions, *cfg.clipping_range)
            oof_predictions += [predictions]
            fold_data['Val RMSLE'] += [root_mean_squared_log_error(y_val, predictions)]
            print(f'Fold {num + 1} Val RMSLE: {fold_data["Val RMSLE"][-1]}')
        oof_predictions = np.concatenate(oof_predictions, axis=0)
        np.save(
            fr'{Configuration.model_base_save_directory[cfg.model_class]}/oof_predictions_{Configuration.num_folds}_fold.npy',
            oof_predictions)
    else:
        model = cfg.model_class(Configuration.model_init_kwarg_mapping[cfg.model_class])
        model.fit(X, y, Configuration.get_default_fit_kwargs(X, model_class=cfg.model_class))

    train_history = pd.DataFrame(fold_data)
    print(f'Average Val RMSLE: {train_history["Val RMSLE"].mean()}')

    if cv_flag:
        for num, model in enumerate(models):
            with open(f'{Configuration.model_base_save_directory[cfg.model_class]}/{cfg.method}_fold_{num + 1}.pickle',
                      'wb') as f:
                pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
        create_submission(X_sub, models)


if __name__ == '__main__':
    train(cfg=TrainConfig())
