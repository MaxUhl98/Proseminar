import pandas as pd
from configuration import Configuration
from sklearn.metrics import root_mean_squared_log_error
from autogluon.core.metrics import make_scorer
def get_autogluon_scorer():
    return make_scorer(name='rmsle', score_func=root_mean_squared_log_error, optimum=0, greater_is_better=False,
                       needs_pred=True)
def run_autogluon_feature_pruning(
        *,
        train_data: pd.DataFrame,target:pd.DataFrame
) -> list[str]:
    from autogluon.core.models import BaggedEnsembleModel
    from autogluon.core.utils.feature_selection import FeatureSelector
    from autogluon.tabular.models.lgb.lgb_model import LGBModel

    # -- Input Data
    train_data = train_data.copy()
    n_features_original = train_data.shape[1] - 1  # -1 for target column
    y = target['Rings']

    # -- Problem Settings
    problem_kwargs = dict(
        problem_type='regression',
        eval_metric=get_autogluon_scorer(),
    )
    time_limit = 60 * 60
    n_sub_sample = 300000

    # Get Proxy Model

    proxy_model = LGBModel
    model_hps = dict(extra_tees=True)

    proxy_model_kwargs = dict(
        **problem_kwargs,
        hyperparameters=dict(
            **model_hps,
        ),
    )
    proxy_model = BaggedEnsembleModel(
        model_base=proxy_model,
        model_base_kwargs=proxy_model_kwargs,
        random_state=42,
    )
    fit_kwargs = dict(
        k_fold=4,
        n_repeats=1,
        replace_bag=False,
    )

    fs = FeatureSelector(
        model=proxy_model,
        time_limit=time_limit,
        problem_type=problem_kwargs["problem_type"],
        seed=Configuration.random_seed,
        raise_exception=True,
    )
    print(train_data.shape, y.shape)
    print(train_data.isna().sum().sum())
    candidate_features = fs.select_features(
        X=train_data,
        y=y,
        n_train_subsample=n_sub_sample,
        # Play around with these variables:
        prune_threshold="noise",
        prune_ratio=0.15,
        stopping_round=4,
        **fit_kwargs,
    )

    print(f"Original #feat: {n_features_original}, #feat after Autogluon Feature Pruning: {len(candidate_features)}")
    print(f"Features:", candidate_features)
    return candidate_features


print(run_autogluon_feature_pruning(train_data=pd.read_feather('data/undummied/X.feather'), target=pd.read_feather('data/undummied/y.feather'), ))
