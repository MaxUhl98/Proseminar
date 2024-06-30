import os.path
import pickle
import pprint

import pandas as pd
from openfe import transform
from typing import *
import openfe as of
from configuration import Configuration
import numpy as np

def load_ofe() -> tuple[Any, Any]:
    with open('feature_generators/ofe_original_feature_generator.pkl', 'rb') as inp:
        ofe = pickle.load(inp)
    with open('feature_generators/ofe_original_features.pkl', 'rb') as inp:
        features = pickle.load(inp)
    return ofe, features


def preprocess_features(train_path: str = 'playground-series-s4e4/train.csv', n_jobs: int = 1,
                        num_engineered_features: int = 10 ** 4, use_additional_data: bool = True,
                        reduce_features: bool = True, dummy_data: bool = True):
    df = pd.read_csv(train_path)
    if use_additional_data:
        abalone_data = pd.read_csv('playground-series-s4e4/abalone.csv')
        y_additional_train = abalone_data.Rings
        X_additional_train = abalone_data.drop(columns='Rings')
        X_additional_train = X_additional_train.rename(
            columns={'Whole_weight': 'Whole weight', 'Shucked weight': 'Whole weight.1',
                     'Viscera weight': 'Whole weight.2',
                     'Shell weight': 'Shell weight'})

    y = df['Rings']
    X = df.drop(columns=['Rings', 'id'])

    if reduce_features:
        features = get_reduced_features(X)
        if not os.path.exists('feature_generators/current_reduced_feature_generator.pkl'):
            features = of.OpenFE().fit(X, y, 'regression', candidate_features_list=features)
            pickle.dump(features, open('feature_generators/current_reduced_feature_generator.pkl', 'wb'))
        else:
            features = pickle.load(open('feature_generators/current_reduced_feature_generator.pkl', 'rb'))
    else:
        features = get_features(X)
        if not os.path.exists('feature_generators/current_feature_generator.pkl'):
            features = of.OpenFE().fit(X, y, 'regression', candidate_features_list=features)
            pickle.dump(features, open('feature_generators/current_feature_generator.pkl', 'wb'))
        else:
            features = pickle.load(open('feature_generators/current_feature_generator.pkl', 'rb'))

    X, _ = transform(X, X.iloc[0:0], features[:num_engineered_features],
                     n_jobs=n_jobs)
    if use_additional_data:
        X_additional_train, _ = transform(X_additional_train, X_additional_train.iloc[0:0],
                                          features[:num_engineered_features],
                                          n_jobs=n_jobs)

    if dummy_data:

        if use_additional_data:
            X_additional_train = pd.get_dummies(X_additional_train, columns=['Sex'])
            X_additional_train = X_additional_train.astype(dtype=float)

        X = pd.get_dummies(X, columns=['Sex'])
        X = X.astype(dtype=float)[list(X_additional_train.columns)] if use_additional_data else X.astype(float)

    if use_additional_data:
        return X, y, X_additional_train, y_additional_train, features
    else:
        return X, y, None, None, features


def get_features(train_data: pd.DataFrame):
    all_features = list(train_data.columns)
    categorical_columns = list(train_data.select_dtypes(include=['category', 'object']).columns)
    ordinal_columns = list(train_data.select_dtypes(include=['int']).columns)

    soft_ordinal = [f for f in all_features if
                    (train_data[f].nunique() <= 100) and (f not in ordinal_columns + categorical_columns)]
    numerical_features = [f for f in all_features if f not in categorical_columns]
    candidate_features = of.get_candidate_features(
        numerical_features=numerical_features,
        categorical_features=categorical_columns,
        ordinal_features=ordinal_columns + soft_ordinal,
        order=1
    )
    return candidate_features


def get_reduced_features(train_data: pd.DataFrame) -> list[of.FeatureSelector.Node]:
    all_features = list(train_data.columns)
    categorical_columns = list(train_data.select_dtypes(include=['category', 'object']).columns)
    ordinal_columns = list(train_data.select_dtypes(include=['int']).columns)

    soft_ordinal = [f for f in all_features if
                    (train_data[f].nunique() <= 100) and (f not in ordinal_columns + categorical_columns)]
    numerical_features = [f for f in all_features if f not in categorical_columns]
    candidate_features = of.get_candidate_features(
        numerical_features=numerical_features,
        categorical_features=categorical_columns,
        ordinal_features=ordinal_columns + soft_ordinal,
        order=1
    )

    # Restrict Search Space of Candidate Features
    candidate_features = [
        f
        for f in candidate_features
        if f.name
           in [
               "freq",
               "round",
               "residual",
               "/",
               "*",
               "GroupByThenMedian",
               "GroupByThenStd",
               "GroupByThenFreq",
               "GroupByThenNUnique",
               "Combine",
               "<p0.2",
               "<p0.4",
               "<p0.6",
               "<p0.8",
           ]
    ]
    return candidate_features


def create_submission(df_submission: pd.DataFrame, models: List[Any]):
    # Save Submission id's to add them to the submission dataframe later (required by the competition for identification)
    submission_ids = df_submission['id']

    X_sub = df_submission.drop(columns='id')
    preds = sum([_model.predict(X_sub.values) for _model in models]) / Configuration.num_folds
    df_submission = pd.DataFrame({'id': submission_ids, 'prediction': preds})
    df_submission.index = df_submission.id
    df_submission.drop(columns='id', inplace=True)
    submission_path = f'submissions/{Configuration.model_base_save_directory[models[0].__class__].rsplit("/", 1)[1]}/submission.csv'
    df_submission.to_csv(submission_path)
    print(f'Created submission at {submission_path}')


def generate_preprocessed_data_files():
    X_submission = pd.read_csv('playground-series-s4e4/test.csv')
    _ids = X_submission.pop('id')
    config_list = [
        {'suffix': '_reduced', 'use_additional_data': True, 'dummy_data': True, 'reduce_features': True},
        {'suffix': '_reduced_raw', 'use_additional_data': False, 'dummy_data': True, 'reduce_features': True},
        {'suffix': '_reduced', 'use_additional_data': True, 'dummy_data': False, 'reduce_features': True},
        {'suffix': '_reduced_raw', 'use_additional_data': False, 'dummy_data': False, 'reduce_features': True},
        {'suffix': '', 'use_additional_data': True, 'dummy_data': True, 'reduce_features': False},
        {'suffix': '_raw', 'use_additional_data': False, 'dummy_data': True, 'reduce_features': False},
        {'suffix': '', 'use_additional_data': True, 'dummy_data': False, 'reduce_features': False},
        {'suffix': '_raw', 'use_additional_data': False, 'dummy_data': False, 'reduce_features': False},
    ]
    for config in config_list:
        X, y, X_extra, y_extra, features = preprocess_features(
            n_jobs=30,
            use_additional_data=config['use_additional_data'],
            dummy_data=config['dummy_data'],
            reduce_features=config['reduce_features']
        )
        suffix = config['suffix']
        base_dir = 'data/dummied' if config['dummy_data'] else 'data/undummied'
        X.to_feather(f'{base_dir}/X{suffix}.feather')
        y.to_frame().to_feather(f'{base_dir}/y{suffix}.feather')
        if config['use_additional_data']:
            X_extra.to_feather(f'{base_dir}/X{suffix}_extra.feather')
            y_extra.to_frame().to_feather(f'{base_dir}/y{suffix}_extra.feather')
        X_sub = transform(X_submission.copy(), X_submission.iloc[0:0].copy(), features, n_jobs=Configuration.n_jobs)[0]
        X_sub = pd.get_dummies(X_sub, columns=['Sex']) if config['dummy_data'] else X_sub
        assert X.shape[1] == X_sub.shape[1], AssertionError(X.shape, X_sub.shape)
        X_sub = pd.concat([X_sub, _ids], axis=1)
        if config['use_additional_data']:
            X_sub.to_feather(f'{base_dir}/X_sub{suffix}_extra.feather')
        else:
            X_sub.to_feather(f'{base_dir}/X_sub{suffix}.feather')


def get_reduced_cols(df:pd.DataFrame, y:pd.DataFrame, cross_correlation_threshold:float=.95,min_target_correlation:float=.5 ):
    df = df.drop(columns='Sex').astype(float)
    df = pd.concat([df, y], axis=1).astype(float).dropna()
    relevant_cols = df.std().loc[df.std() >= .001].index
    df = df[relevant_cols]
    drops = ['autoFE_f_173', 'autoFE_f_174', 'autoFE_f_187']
    df.drop(columns=drops, inplace=True)
    corr = df.corr()
    idx, cols = corr.index, list(corr.columns)
    data = pd.DataFrame(np.tril(corr, -1), index=idx, columns=cols)
    data = np.abs(data)
    data.iloc[-1, -1] = .5
    drops = data.loc[(data.iloc[-1] <= min_target_correlation)].index
    data.drop(columns=drops, inplace=True)
    data.drop(drops, inplace=True)
    cols = list(data.columns)
    while data.max().max() >= cross_correlation_threshold:
        col = data.max().loc[data.max() == data.max().max()].index
        if len(col) > 1:
            col = col[0]
        col2 = cols[data[col].values.argmax()]
        if corr[col].loc['Rings'].item() >= corr[col2].loc['Rings'].item():
            cols.remove(col2)
            data.drop(columns=col2, inplace=True)
            data.drop(col2, inplace=True)
        else:
            cols.remove(col)
            data.drop(columns=col, inplace=True)
            data.drop(col, inplace=True)
    return cols
