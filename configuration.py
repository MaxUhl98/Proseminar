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

    selected_features = ['Shell weight', 'autoFE_f_0', 'autoFE_f_1', 'autoFE_f_2', 'autoFE_f_3', 'autoFE_f_4', 'autoFE_f_5', 'autoFE_f_6', 'autoFE_f_7', 'autoFE_f_8', 'autoFE_f_10', 'autoFE_f_11', 'autoFE_f_12', 'autoFE_f_14', 'autoFE_f_15', 'autoFE_f_17', 'autoFE_f_18', 'autoFE_f_19', 'autoFE_f_20', 'autoFE_f_23', 'autoFE_f_24', 'autoFE_f_26', 'autoFE_f_29', 'autoFE_f_30', 'autoFE_f_41', 'autoFE_f_42', 'autoFE_f_43', 'autoFE_f_44', 'autoFE_f_45', 'autoFE_f_46', 'autoFE_f_47', 'autoFE_f_48', 'autoFE_f_49', 'autoFE_f_50', 'autoFE_f_51', 'autoFE_f_52', 'autoFE_f_53', 'autoFE_f_54', 'autoFE_f_55', 'autoFE_f_56', 'autoFE_f_57', 'autoFE_f_58', 'autoFE_f_59', 'autoFE_f_60', 'autoFE_f_61', 'autoFE_f_62', 'autoFE_f_63', 'autoFE_f_64', 'autoFE_f_65', 'autoFE_f_66', 'autoFE_f_67', 'autoFE_f_68', 'autoFE_f_69', 'autoFE_f_70', 'autoFE_f_71', 'autoFE_f_72', 'autoFE_f_73', 'autoFE_f_74', 'autoFE_f_75', 'autoFE_f_76', 'autoFE_f_77', 'autoFE_f_78', 'autoFE_f_79', 'autoFE_f_80', 'autoFE_f_81', 'autoFE_f_82', 'autoFE_f_83', 'autoFE_f_84', 'autoFE_f_85', 'autoFE_f_86', 'autoFE_f_87', 'autoFE_f_88', 'autoFE_f_89', 'autoFE_f_90', 'autoFE_f_91', 'autoFE_f_92', 'autoFE_f_93', 'autoFE_f_94', 'autoFE_f_95', 'autoFE_f_96', 'autoFE_f_97', 'autoFE_f_98', 'autoFE_f_99', 'autoFE_f_100', 'autoFE_f_101', 'autoFE_f_102', 'autoFE_f_103', 'autoFE_f_104', 'autoFE_f_105', 'autoFE_f_106', 'autoFE_f_107', 'autoFE_f_108', 'autoFE_f_109', 'autoFE_f_110', 'autoFE_f_111', 'autoFE_f_112', 'autoFE_f_113', 'autoFE_f_114', 'autoFE_f_115', 'autoFE_f_116', 'autoFE_f_117', 'autoFE_f_118', 'autoFE_f_119', 'autoFE_f_120', 'autoFE_f_121', 'autoFE_f_122', 'autoFE_f_123', 'autoFE_f_124', 'autoFE_f_125', 'autoFE_f_126', 'autoFE_f_127', 'autoFE_f_128', 'autoFE_f_129', 'autoFE_f_130', 'autoFE_f_131', 'autoFE_f_132', 'autoFE_f_133', 'autoFE_f_134', 'autoFE_f_135', 'autoFE_f_136', 'autoFE_f_137', 'autoFE_f_138', 'autoFE_f_139', 'autoFE_f_140', 'autoFE_f_141', 'autoFE_f_142', 'autoFE_f_143', 'autoFE_f_144', 'autoFE_f_145', 'autoFE_f_146', 'autoFE_f_147', 'autoFE_f_148', 'autoFE_f_149', 'autoFE_f_150', 'autoFE_f_151', 'autoFE_f_152', 'autoFE_f_153', 'autoFE_f_154', 'autoFE_f_155', 'autoFE_f_156', 'autoFE_f_157', 'autoFE_f_158', 'autoFE_f_159', 'autoFE_f_160', 'autoFE_f_161', 'autoFE_f_162', 'autoFE_f_163', 'autoFE_f_164', 'autoFE_f_165', 'autoFE_f_166', 'autoFE_f_167', 'autoFE_f_168', 'autoFE_f_169', 'autoFE_f_170', 'autoFE_f_171', 'autoFE_f_172', 'autoFE_f_173', 'autoFE_f_174', 'autoFE_f_175', 'autoFE_f_176', 'autoFE_f_177', 'autoFE_f_178', 'autoFE_f_179', 'autoFE_f_180', 'autoFE_f_181', 'autoFE_f_182', 'autoFE_f_183', 'autoFE_f_184', 'autoFE_f_185', 'autoFE_f_186', 'autoFE_f_187', 'autoFE_f_188', 'autoFE_f_189', 'autoFE_f_190', 'autoFE_f_191', 'autoFE_f_192', 'autoFE_f_193', 'autoFE_f_194', 'autoFE_f_195']


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
