from tests.shared_fixtures import test_sample as test_sample_data
from yieldengine.loading.sample import Sample
from yieldengine.preprocessing.cross_validation import CircularCrossValidator
from yieldengine.modeling.model_selector import ModelSelector
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from lightgbm.sklearn import LGBMRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error


def test_model_selector(test_sample_data):
    # drop columns that should not take part in modeling
    test_sample_data = test_sample_data.drop(columns=["Date", "Batch Id"])

    # replace values of +/- infinite with n/a, then drop all n/a columns:
    test_sample_data = test_sample_data.replace([np.inf, -np.inf], np.nan).dropna(
        axis=1, how="all"
    )

    feature_columns = list(test_sample_data.columns)
    feature_columns.remove("Yield")

    sample = Sample(sample=test_sample_data, target="Yield", features=feature_columns)

    # define the circular cross validator with just 5 folds (to speed up testing)
    circular_cv = CircularCrossValidator(
        num_samples=len(sample), test_ratio=0.20, num_folds=5
    )

    # define a ColumnTransformer to pre-process:
    preprocessor = ColumnTransformer(
        [
            ("numerical", SimpleImputer(strategy="mean"), sample.numerical_features),
            (
                "categorical",
                OneHotEncoder(sparse=False, handle_unknown="ignore"),
                sample.categorical_features,
            ),
        ]
    )

    # define a sklearn Pipeline, containing the preprocessor defined above:
    pre_pipeline = Pipeline([("prep", preprocessor)])

    # run fit_transform once to assure it works:
    pre_pipeline.fit_transform(sample.feature_data)

    # define all grid-searchers, keep in this list:
    searchers = []

    models_and_parameters = [
        (
            LGBMRegressor(),
            {
                "max_depth": (5, 10),
                "min_split_gain": (0.1, 0.2),
                "num_leaves": (50, 100, 200),
            },
        ),
        (AdaBoostRegressor(), {"n_estimators": (50, 80)}),
        (RandomForestRegressor(), {"n_estimators": (50, 80)}),
        (
            DecisionTreeRegressor(),
            {"max_depth": (0.5, 1.0), "max_features": (0.5, 1.0)},
        ),
        (ExtraTreeRegressor(), {"max_depth": (5, 10, 12)}),
        (SVR(), {"gamma": (0.5, 1), "C": (50, 100)}),
        (LinearRegression(), {"normalize": (False, True)}),
    ]

    for model, parameters in models_and_parameters:
        search = GridSearchCV(
            estimator=model,
            cv=circular_cv,
            param_grid=parameters,
            scoring=make_scorer(mean_squared_error, greater_is_better=False),
            n_jobs=-1,
        )
        searchers.append(search)

    # instantiate the model selector
    ms = ModelSelector(searchers=searchers, preprocessing=pre_pipeline)

    # retrieve a pipeline
    complete_pipeline = ms.construct_pipeline()

    # train the models
    complete_pipeline.fit(sample.feature_data, sample.target_data)

    # when done, get ranking
    ranked_models = ms.rank_models()
    # check types
    assert type(ranked_models) == list
    assert type(ranked_models[0]) == GridSearchCV
    # check sorting
    assert (
        ranked_models[0].best_score_
        >= ranked_models[1].best_score_
        >= ranked_models[2].best_score_
    )

    # print summary:
    print("Ranked models:")
    print(ms.summary_string())

    # get ranked model-instances:
    ranked_model_instances = ms.rank_model_instances(n_best_ranked=3)

    # check data structure
    assert type(ranked_model_instances) == list
    assert type(ranked_model_instances[0]) == dict
    assert {"score", "estimator", "params"} == ranked_model_instances[0].keys()

    # check sorting
    assert (
        ranked_model_instances[0]["score"]
        >= ranked_model_instances[1]["score"]
        >= ranked_model_instances[2]["score"]
    )

    # test transform():
    assert complete_pipeline.transform(sample.feature_data).shape == (
        len(sample),
        len(searchers),
    )


def test_model_selector_no_preprocessing():
    # filter out warnings triggerd by sk-learn/numpy
    import warnings

    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
    warnings.filterwarnings("ignore", message="You are accessing a training score")
    from sklearn import datasets, svm

    # load example data
    iris = datasets.load_iris()

    # define a yield-engine circular CV:
    my_cv = CircularCrossValidator(
        num_samples=len(iris.data), test_ratio=0.21, num_folds=50
    )

    # define parameters and model
    parameters = {"kernel": ("linear", "rbf"), "C": [1, 10]}
    svc = svm.SVC(gamma="scale")

    # use the defined my_cv circular CV within GridSearchCV:
    clf = GridSearchCV(svc, parameters, cv=my_cv)

    ms = ModelSelector(searchers=[clf])
    p = ms.construct_pipeline()
    p.fit(iris.data, iris.target)
    print(pd.DataFrame(ms.rank_model_instances()).head())
