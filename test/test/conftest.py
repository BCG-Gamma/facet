import functools
import logging
import operator
from typing import Any, List, Mapping, Optional, Sequence, Set

import numpy as np
import pandas as pd
import pytest
from sklearn import datasets
from sklearn.model_selection import BaseCrossValidator, KFold
from sklearn.utils import Bunch

from sklearndf import TransformerDF
from sklearndf.pipeline import RegressorPipelineDF
from sklearndf.regression import (
    SVRDF,
    AdaBoostRegressorDF,
    DecisionTreeRegressorDF,
    ExtraTreeRegressorDF,
    LinearRegressionDF,
    RandomForestRegressorDF,
)
from sklearndf.regression.extra import LGBMRegressorDF
from sklearndf.transformation import (
    ColumnTransformerDF,
    OneHotEncoderDF,
    SimpleImputerDF,
)

import facet
from facet.crossfit import LearnerCrossfit
from facet.data import Sample
from facet.inspection import LearnerInspector, TreeExplainerFactory
from facet.selection import LearnerEvaluation, LearnerGrid, LearnerRanker
from facet.validation import BootstrapCV, StratifiedBootstrapCV

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

# print the FACET logo
print(facet.__logo__)

# disable SHAP debugging messages
logging.getLogger("shap").setLevel(logging.WARNING)

# configure pandas text output
pd.set_option("display.width", None)  # get display width from terminal
pd.set_option("precision", 3)  # 3 digits precision for easier readability

K_FOLDS = 5
N_BOOTSTRAPS = 30

STEP_IMPUTE = "impute"
STEP_ONE_HOT_ENCODE = "one-hot-encode"


@pytest.fixture
def boston_target() -> str:
    return "price"


@pytest.fixture
def iris_target_name() -> str:
    return "species"


@pytest.fixture
def n_jobs() -> int:
    return -1


@pytest.fixture
def cv_kfold() -> KFold:
    # define a CV
    return KFold(n_splits=K_FOLDS)


@pytest.fixture
def cv_bootstrap() -> BaseCrossValidator:
    # define a CV
    return BootstrapCV(n_splits=N_BOOTSTRAPS, random_state=42)


@pytest.fixture
def cv_stratified_bootstrap() -> BaseCrossValidator:
    # define a CV
    return StratifiedBootstrapCV(n_splits=N_BOOTSTRAPS, random_state=42)


@pytest.fixture
def regressor_grids(simple_preprocessor: TransformerDF) -> List[LearnerGrid]:
    random_state = {"random_state": [42]}

    return [
        LearnerGrid(
            pipeline=RegressorPipelineDF(
                preprocessing=simple_preprocessor, regressor=LGBMRegressorDF()
            ),
            learner_parameters={
                "max_depth": [5, 10],
                "min_split_gain": [0.1, 0.2],
                "num_leaves": [50, 100, 200],
                **random_state,
            },
        ),
        LearnerGrid(
            pipeline=RegressorPipelineDF(
                preprocessing=simple_preprocessor, regressor=AdaBoostRegressorDF()
            ),
            learner_parameters={"n_estimators": [50, 80], **random_state},
        ),
        LearnerGrid(
            pipeline=RegressorPipelineDF(
                preprocessing=simple_preprocessor, regressor=RandomForestRegressorDF()
            ),
            learner_parameters={"n_estimators": [50, 80], **random_state},
        ),
        LearnerGrid(
            pipeline=RegressorPipelineDF(
                preprocessing=simple_preprocessor, regressor=DecisionTreeRegressorDF()
            ),
            learner_parameters={
                "max_depth": [0.5, 1.0],
                "max_features": [0.5, 1.0],
                **random_state,
            },
        ),
        LearnerGrid(
            pipeline=RegressorPipelineDF(
                preprocessing=simple_preprocessor, regressor=ExtraTreeRegressorDF()
            ),
            learner_parameters={"max_depth": [5, 10, 12], **random_state},
        ),
        LearnerGrid(
            pipeline=RegressorPipelineDF(
                preprocessing=simple_preprocessor, regressor=SVRDF()
            ),
            learner_parameters={"gamma": [0.5, 1], "C": [50, 100]},
        ),
        LearnerGrid(
            pipeline=RegressorPipelineDF(
                preprocessing=simple_preprocessor, regressor=LinearRegressionDF()
            ),
            learner_parameters={"normalize": [False, True]},
        ),
    ]


@pytest.fixture
def regressor_ranker(
    cv_kfold: KFold,
    regressor_grids: List[LearnerGrid[RegressorPipelineDF]],
    sample: Sample,
    n_jobs: int,
) -> LearnerRanker[RegressorPipelineDF]:
    return LearnerRanker(
        grids=regressor_grids, cv=cv_kfold, scoring="r2", n_jobs=n_jobs
    ).fit(sample=sample)


@pytest.fixture
def best_lgbm_crossfit(
    regressor_ranker: LearnerRanker[RegressorPipelineDF],
    cv_kfold: KFold,
    sample: Sample,
    n_jobs: int,
) -> LearnerCrossfit[RegressorPipelineDF]:
    # we get the best model_evaluation which is a LGBM - for the sake of test
    # performance
    best_lgbm_evaluation: LearnerEvaluation[RegressorPipelineDF] = [
        evaluation
        for evaluation in regressor_ranker.ranking_
        if isinstance(evaluation.pipeline.regressor, LGBMRegressorDF)
    ][0]

    best_lgbm_regressor: RegressorPipelineDF = best_lgbm_evaluation.pipeline

    return LearnerCrossfit(
        pipeline=best_lgbm_regressor,
        cv=cv_kfold,
        random_state=42,
        n_jobs=n_jobs,
    ).fit(sample=sample)


@pytest.fixture
def feature_names(best_lgbm_crossfit: LearnerCrossfit[RegressorPipelineDF]) -> Set[str]:
    """
    all unique features across the models in the crossfit, after preprocessing
    """
    return functools.reduce(
        operator.or_,
        (set(model.feature_names_out_) for model in best_lgbm_crossfit.models()),
    )


@pytest.fixture
def regressor_inspector(
    best_lgbm_crossfit: LearnerCrossfit[RegressorPipelineDF], n_jobs: int
) -> LearnerInspector:
    inspector = LearnerInspector(
        explainer_factory=TreeExplainerFactory(
            feature_perturbation="tree_path_dependent", use_background_dataset=True
        ),
        n_jobs=n_jobs,
    ).fit(crossfit=best_lgbm_crossfit)

    # disable legacy calculations; we used them in the constructor so the legacy
    # SHAP decomposer is created along with the new SHAP vector projector
    inspector._legacy = False

    return inspector


@pytest.fixture
def simple_preprocessor(sample: Sample) -> TransformerDF:
    features = sample.features

    column_transforms = []

    numeric_columns = features.select_dtypes(np.number).columns
    if numeric_columns is not None and len(numeric_columns) > 0:
        column_transforms.append(
            (STEP_IMPUTE, SimpleImputerDF(strategy="median"), numeric_columns)
        )

    category_columns = features.select_dtypes(object).columns
    if category_columns is not None and len(category_columns) > 0:
        column_transforms.append(
            (
                STEP_ONE_HOT_ENCODE,
                OneHotEncoderDF(sparse=False, handle_unknown="ignore"),
                category_columns,
            )
        )

    return ColumnTransformerDF(transformers=column_transforms)


@pytest.fixture
def boston_df(boston_target: str) -> pd.DataFrame:
    #  load sklearn test-data and convert to pd
    boston: Bunch = datasets.load_boston()

    return pd.DataFrame(
        data=np.c_[boston.data, boston.target],
        columns=[*boston.feature_names, boston_target],
    )


@pytest.fixture
def sample(boston_df: pd.DataFrame, boston_target: str) -> Sample:
    return Sample(observations=boston_df.iloc[:100, :], target_name=boston_target)


@pytest.fixture
def iris_df(iris_target_name: str) -> pd.DataFrame:
    #  load sklearn test-data and convert to pd
    iris: Bunch = datasets.load_iris()

    iris_df = pd.DataFrame(
        data=np.c_[iris.data, iris.target],
        columns=[*iris.feature_names, iris_target_name],
    )

    # replace target numericals with actual class labels
    iris_df.loc[:, iris_target_name] = (
        iris_df.loc[:, iris_target_name]
        .astype(int)
        .map(dict(enumerate(iris.target_names)))
    )

    return iris_df


@pytest.fixture
def iris_sample(iris_df: pd.DataFrame, iris_target_name: str) -> Sample:
    # the iris dataset
    return Sample(
        observations=iris_df.assign(weight=2.0),
        target_name=iris_target_name,
        weight_name="weight",
    )


@pytest.fixture
def iris_sample_binary(iris_sample: Sample) -> Sample:
    # the iris dataset, retaining only two categories so we can do binary classification
    return iris_sample.subsample(
        loc=iris_sample.target.isin(["virginica", "versicolor"])
    )


@pytest.fixture
def iris_sample_binary_dual_target(
    iris_sample_binary: Sample, iris_target_name
) -> Sample:
    # the iris dataset, retaining only two categories so we can do binary classification
    target = pd.Series(
        index=iris_sample_binary.index,
        data=pd.Categorical(iris_sample_binary.target).codes,
        name=iris_target_name,
    )
    iris_target_2 = f"{iris_target_name}2"
    return Sample(
        iris_sample_binary.features.join(target).join(target.rename(iris_target_2)),
        target_name=[iris_sample_binary.target_name, iris_target_2],
    )


def check_ranking(
    ranking: List[LearnerEvaluation],
    expected_scores: Sequence[float],
    expected_learners: Optional[Sequence[type]],
    expected_parameters: Optional[Mapping[int, Mapping[str, Any]]],
) -> None:
    """
    Test helper to check rankings produced by learner rankers

    :param ranking: a list of LearnerEvaluations
    :param expected_scores: expected ranking scores, rounded to 3 decimal places
    :param expected_learners: expected learner classes
    :param expected_parameters: expected learner parameters
    :return: None
    """

    if expected_learners is None:
        expected_learners = [None] * len(ranking)

    for rank, (learner_eval, score_expected, learner_expected) in enumerate(
        zip(ranking, expected_scores, expected_learners)
    ):
        score_actual = round(learner_eval.ranking_score, 3)
        assert score_actual == pytest.approx(score_expected, abs=0.1), (
            f"unexpected score for learner at rank #{rank + 1}: "
            f"got {score_actual} but expected {score_expected}"
        )
        if learner_expected is not None:
            learner_actual = learner_eval.pipeline.final_estimator
            assert type(learner_actual) == learner_expected, (
                f"unexpected class for learner at rank #{rank}: "
                f"got {type(learner_actual)} but expected {learner_expected}"
            )

    if expected_parameters is not None:
        for rank, parameters_expected in expected_parameters.items():
            parameters_actual = ranking[rank].parameters
            assert parameters_actual == parameters_expected, (
                f"unexpected parameters for learner at rank #{rank}: "
                f"got {parameters_actual} but expected {parameters_expected}"
            )
