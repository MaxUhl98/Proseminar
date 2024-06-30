from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, StackingRegressor
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold, StratifiedGroupKFold
from typing import *
import pandas as pd


def msle_loss(y_true, y_pred):
    y_pred = np.maximum(y_pred, -1 + 1e-6)
    return ((np.log1p(y_pred) - np.log1p(y_true)) / (1 + y_pred),
            (1 - np.log1p(y_pred) + np.log1p(y_true)) / (1 + y_pred) ** 2)


class Configuration:

    dummied_data_dir:str = 'data/dummied'
    undummied_data_dir:str = 'data/undummied'
    num_folds: int = 10
    random_seed: int = 1
    folder: Union[StratifiedKFold, KFold, GroupKFold, StratifiedGroupKFold] = StratifiedKFold(num_folds, shuffle=True,
                                                                                              random_state=random_seed)
    n_jobs: int = 30
    model_init_kwarg_mapping = {XGBRegressor: {'objective': 'reg:squaredlogerror'},
                                LGBMRegressor: {'objective': msle_loss, 'n_jobs': n_jobs},
                                HistGradientBoostingRegressor: {'max_iter': 10 ** 4},
                                Ridge: {},
                                StackingRegressor: {'estimators': [
                                    ('xgb', XGBRegressor()), ('lgbm', LGBMRegressor()), ('cat', CatBoostRegressor()),
                                    ('hist', HistGradientBoostingRegressor())], 'cv': folder, 'n_jobs': n_jobs,
                                    'final_estimator': Ridge()},
                                CatBoostRegressor: {}
                                }

    model_base_save_directory = {XGBRegressor: r'D:/models/Abalone/models/xgb',
                                 LGBMRegressor: r'D:/models/Abalone/models/lgbm',
                                 HistGradientBoostingRegressor: r'D:/models/Abalone/models/histgb',
                                 Ridge: r'D:/models/Abalone/models/ridge',
                                 StackingRegressor: r'D:/models/Abalone/models/stacking_regressor',
                                 CatBoostRegressor: r'D:/models/Abalone/models/cat'
                                 }

    log_transform_models = [CatBoostRegressor, StackingRegressor, HistGradientBoostingRegressor, Ridge]

    dummy_models = [StackingRegressor, LGBMRegressor, Ridge, XGBRegressor]

    automatic_cv_models = [StackingRegressor]

    @staticmethod
    def get_default_fit_kwargs(X: pd.DataFrame, model_class: Any):
        default_fit_kwargs = {
            XGBRegressor: {},
            LGBMRegressor: {},
            HistGradientBoostingRegressor: {},
            CatBoostRegressor: {'cat_features': list(X.select_dtypes(include=['object', 'category']).columns)},
            Ridge: {},
            StackingRegressor: {}
        }
        return default_fit_kwargs[model_class]
