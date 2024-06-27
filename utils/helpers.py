import pickle
import pandas as pd
from openfe import transform
from typing import *
import openfe as of


def load_ofe() -> tuple[Any, Any]:
    with open('feature_generators/ofe_original_feature_generator.pkl', 'rb') as inp:
        ofe = pickle.load(inp)
    with open('feature_generators/ofe_original_features.pkl', 'rb') as inp:
        features = pickle.load(inp)
    return ofe, features


def preprocess_features(train_path: str = 'playground-series-s4e4/train.csv', n_jobs: int = 1,
                        num_engineered_features: int = 10 ** 4, use_additional_data: bool = True):
    df = pd.read_csv(train_path)
    if use_additional_data:
        abalone_data = pd.read_csv('playground-series-s4e4/abalone.csv')
        y_additional_train = abalone_data.Rings
        X_additional_train = abalone_data.drop(columns='Rings')
        X_additional_train = X_additional_train.rename(
            columns={'Whole_weight': 'Whole weight', 'Shucked weight': 'Whole weight.1',
                     'Viscera weight': 'Whole weight.2',
                     'Shell weight': 'Shell weight'})

    ofe, features = load_ofe()

    y = df['Rings']
    X = df.drop(columns=['Rings', 'id'])
    X, _ = transform(X, X.iloc[0:0], features[:num_engineered_features],
                     n_jobs=n_jobs)
    if use_additional_data:
        X_additional_train, _ = transform(X_additional_train, X_additional_train.iloc[0:0],
                                          features[:num_engineered_features],
                                          n_jobs=n_jobs)
        X_additional_train = pd.get_dummies(X_additional_train)

    X = pd.get_dummies(X)
    X = X.astype(dtype=float)[list(X_additional_train.columns)] if use_additional_data else X.astype(float)
    if use_additional_data:
        X_additional_train = X_additional_train.astype(dtype=float)
        return X, y, X_additional_train, y_additional_train
    else:
        return X, y, None, None


def get_features(train_data: pd.DataFrame) -> list[of.FeatureSelector.Node]:
    all_features = list(train_data.columns)
    categorical_columns = list(train_data.select_dtypes(include=['category', 'object']).columns)
    ordinal_columns = list(train_data.select_dtypes(include=['int']).columns)

    soft_ordinal = [f for f in all_features if (train_data[f].nunique() <= 100) and (f not in ordinal_columns)]
    numerical_features = [f for f in all_features if f not in categorical_columns]
    candidate_features = of.get_candidate_features(
        numerical_features=numerical_features,
        categorical_features=categorical_columns,
        ordinal_features=ordinal_columns + soft_ordinal,
        order=1,  # 2 is likely impossible to use w/o time estimate.
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
               # New Generators
               #   - Hacked into OpenFE by adding `new_data = int(d < d.quantile(X).max())` to the generator options.
               "<p0.2",  # X = 0.2
               "<p0.4",
               "<p0.6",
               "<p0.8",
           ]
    ]
    return candidate_features
