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
from sklearn.feature_selection import SequentialFeatureSelector, VarianceThreshold
from typing import *
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression, RidgeCV, Ridge
from catboost import CatBoostRegressor
from configuration import Configuration
import pickle
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

from sklearn.metrics import make_scorer


class SelectionConfig:
    method: str = 'sfs'
    num_engineered_features: int = 10 ** 4
    use_additional_data: bool = True
    select_features: bool = True
    reduce_features: bool = False
    model_class = LGBMRegressor
    cov_threshold: float = .99
    min_var_threshold: float = .01
    tol = 10 ** -5


def get_model_init_kwargs(cfg: SelectionConfig) -> Dict[str, Any]:
    return Configuration.model_init_kwarg_mapping[cfg.model_class]


def need_log_transform(cfg: SelectionConfig) -> bool:
    if cfg.model_class in Configuration.log_transform_models:
        return True
    return False


def need_dummies(cfg: SelectionConfig) -> bool:
    return True if cfg.model_class in Configuration.dummy_models else False


def need_cv(cfg: SelectionConfig) -> bool:
    return False if cfg.model_class in Configuration.automatic_cv_models else True


def load_data(cfg: SelectionConfig, dummy_flag: bool) -> tuple[pd.DataFrame, pd.DataFrame, Any, Any, pd.DataFrame]:
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


def remove_collinear_features(x, threshold):
    '''
    Objective:
        Remove collinear features in a dataframe with a correlation coefficient
        greater than the threshold. Removing collinear features can help a model 
        to generalize and improves the interpretability of the model.

    Inputs: 
        x: features dataframe
        threshold: features with correlations greater than this value are removed

    Output: 
        dataframe that contains only the non-highly-collinear features
    '''

    # Calculate the correlation matrix
    corr_matrix = x.corr()
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []

    # Iterate through the correlation matrix and compare correlations
    for i in iters:
        for j in range(i + 1):
            item = corr_matrix.iloc[j:(j + 1), (i + 1):(i + 2)]
            col = item.columns
            row = item.index
            val = abs(item.values)

            # If correlation exceeds the threshold
            if val >= threshold:
                # Print the correlated features and the correlation value
                # print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
                drop_cols.append(col.values[0])

    # Drop one of each pair of correlated columns
    drops = set(drop_cols)
    x = x.drop(columns=drops)
    print('Removed Columns {}'.format(drops))
    return x


def select_features(cfg: SelectionConfig):
    rmsle_scorer = make_scorer(root_mean_squared_log_error, greater_is_better=False)
    X, y, X_extra, y_extra, X_sub = load_data(cfg, need_dummies(cfg))

    thresh = VarianceThreshold(cfg.min_var_threshold)
    cols = X.columns
    thresh.fit(X, y)
    X = X[thresh.get_feature_names_out(cols)]
    X = remove_collinear_features(X, cfg.cov_threshold)
    y = y.values.ravel()
    selector = SequentialFeatureSelector(cfg.model_class(**Configuration.model_init_kwarg_mapping[cfg.model_class]),
                                         tol=cfg.tol, scoring=rmsle_scorer, n_jobs=Configuration.n_jobs,
                                         cv=Configuration.num_folds)
    selector.fit_transform(X, y, **Configuration.get_default_fit_kwargs(X, cfg.model_class))
    selected_features = selector.get_feature_names_out(X.columns)
    with open(f'{Configuration.model_base_save_directory[cfg.model_class]}/selected_features_{cfg.method}.txt', 'w',
              encoding='utf-8') as f:
        f.write(str(selected_features))


if __name__ == '__main__':
    select_features(cfg=SelectionConfig())
