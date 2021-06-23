"""
Model inspector tests.
"""
import logging
import warnings
from typing import List, Sequence, Set, TypeVar

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal
from sklearn.datasets import make_classification
from sklearn.model_selection import BaseCrossValidator, KFold

from pytools.viz.dendrogram import DendrogramDrawer, DendrogramReportStyle
from sklearndf import TransformerDF
from sklearndf.classification import (
    GradientBoostingClassifierDF,
    RandomForestClassifierDF,
)
from sklearndf.pipeline import ClassifierPipelineDF, RegressorPipelineDF

from ..conftest import check_ranking
from facet.crossfit import LearnerCrossfit
from facet.data import Sample
from facet.inspection import (
    KernelExplainerFactory,
    LearnerInspector,
    TreeExplainerFactory,
)
from facet.selection import LearnerGrid, LearnerRanker
from facet.validation import BootstrapCV, StratifiedBootstrapCV

# noinspection PyMissingOrEmptyDocstring

log = logging.getLogger(__name__)

T = TypeVar("T")


@pytest.fixture
def iris_classifier_ranker_binary(
    iris_sample_binary: Sample,
    cv_stratified_bootstrap: StratifiedBootstrapCV,
    n_jobs: int,
) -> LearnerRanker[ClassifierPipelineDF[RandomForestClassifierDF]]:
    return fit_learner_ranker(
        sample=iris_sample_binary, cv=cv_stratified_bootstrap, n_jobs=n_jobs
    )


@pytest.fixture
def iris_classifier_ranker_multi_class(
    iris_sample: Sample, cv_stratified_bootstrap: StratifiedBootstrapCV, n_jobs: int
) -> LearnerRanker[ClassifierPipelineDF[RandomForestClassifierDF]]:
    return fit_learner_ranker(
        sample=iris_sample, cv=cv_stratified_bootstrap, n_jobs=n_jobs
    )


@pytest.fixture
def iris_classifier_ranker_dual_target(
    iris_sample_binary_dual_target: Sample, cv_bootstrap: BootstrapCV, n_jobs: int
) -> LearnerRanker[ClassifierPipelineDF[RandomForestClassifierDF]]:
    return fit_learner_ranker(
        sample=iris_sample_binary_dual_target, cv=cv_bootstrap, n_jobs=n_jobs
    )


@pytest.fixture
def iris_classifier_crossfit_binary(
    iris_classifier_ranker_binary: LearnerRanker[ClassifierPipelineDF],
) -> LearnerCrossfit[ClassifierPipelineDF[RandomForestClassifierDF]]:
    return iris_classifier_ranker_binary.best_model_crossfit_


@pytest.fixture
def iris_classifier_crossfit_multi_class(
    iris_classifier_ranker_multi_class: LearnerRanker[ClassifierPipelineDF],
) -> LearnerCrossfit[ClassifierPipelineDF[RandomForestClassifierDF]]:
    return iris_classifier_ranker_multi_class.best_model_crossfit_


@pytest.fixture
def iris_inspector_multi_class(
    iris_classifier_crossfit_multi_class: LearnerCrossfit[
        ClassifierPipelineDF[RandomForestClassifierDF]
    ],
    n_jobs: int,
) -> LearnerInspector[ClassifierPipelineDF[RandomForestClassifierDF]]:
    return LearnerInspector(shap_interaction=True, n_jobs=n_jobs).fit(
        crossfit=iris_classifier_crossfit_multi_class
    )


def test_model_inspection(
    regressor_grids: Sequence[LearnerGrid[RegressorPipelineDF]],
    regressor_ranker: LearnerRanker[RegressorPipelineDF],
    best_lgbm_crossfit: LearnerCrossfit[RegressorPipelineDF],
    feature_names: Set[str],
    regressor_inspector: LearnerInspector,
    cv_kfold: KFold,
    sample: Sample,
    simple_preprocessor: TransformerDF,
    n_jobs: int,
) -> None:

    # define checksums for this test
    expected_scores = [0.418, 0.400, 0.386, 0.385, 0.122] + [
        0.122,
        -0.074,
        -0.074,
        -0.074,
        -0.074,
    ]

    log.debug(f"\n{regressor_ranker.summary_report()}")

    check_ranking(
        ranking=regressor_ranker.ranking_,
        expected_scores=expected_scores,
        expected_learners=None,
        expected_parameters=None,
    )

    # using an invalid consolidation method raises an exception
    with pytest.raises(ValueError, match="unknown aggregation method: invalid"):
        regressor_inspector.shap_values(aggregation="invalid")

    shap_values_raw: pd.DataFrame = regressor_inspector.shap_values(aggregation=None)
    shap_values_mean = regressor_inspector.shap_values(
        aggregation=LearnerInspector.AGG_MEAN
    )
    shap_values_std = regressor_inspector.shap_values(
        aggregation=LearnerInspector.AGG_STD
    )

    # method shap_values without parameter is equal to "mean" consolidation
    assert_frame_equal(shap_values_mean, regressor_inspector.shap_values())

    # the length of rows in shap_values should be equal to the unique observation
    # indices we have had in the predictions_df
    assert len(shap_values_mean) == len(sample)

    # index names
    assert shap_values_mean.index.names == [Sample.IDX_OBSERVATION]
    assert shap_values_mean.columns.names == [Sample.IDX_FEATURE]
    assert shap_values_std.index.names == [Sample.IDX_OBSERVATION]
    assert shap_values_std.columns.names == [Sample.IDX_FEATURE]
    assert shap_values_raw.index.names == ["split", "observation"]
    assert shap_values_raw.columns.names == [Sample.IDX_FEATURE]

    # column index
    assert set(shap_values_mean.columns) == feature_names

    # check that the SHAP values add up to the predictions
    shap_totals_raw = shap_values_raw.sum(axis=1)

    for split_id, model in enumerate(best_lgbm_crossfit.models()):
        # for each model in the crossfit, calculate the difference between total
        # SHAP values and prediction for every observation. This is always the same
        # constant value, so `mad` (mean absolute deviation) is zero

        shap_minus_pred = shap_totals_raw.xs(key=split_id) - model.predict(
            X=sample.features
        )
        assert (
            round(shap_minus_pred.mad(), 12) == 0.0
        ), f"predictions matching total SHAP for split {split_id}"

    #  test the ModelInspector with a KernelExplainer:

    inspector_2 = LearnerInspector(
        explainer_factory=KernelExplainerFactory(link="identity", data_size_limit=20),
        n_jobs=n_jobs,
    ).fit(crossfit=best_lgbm_crossfit)
    inspector_2.shap_values()

    linkage_tree = inspector_2.feature_association_linkage()

    print()
    DendrogramDrawer(style="text").draw(data=linkage_tree, title="Test")


def test_binary_classifier_ranking(iris_classifier_ranker_binary) -> None:

    expected_learner_scores = [0.872, 0.868, 0.866, 0.859]

    log.debug(f"\n{iris_classifier_ranker_binary.summary_report()}")
    check_ranking(
        ranking=iris_classifier_ranker_binary.ranking_,
        expected_scores=expected_learner_scores,
        expected_learners=[RandomForestClassifierDF] * 4,
        expected_parameters={
            2: dict(classifier__min_samples_leaf=4, classifier__n_estimators=10),
            3: dict(classifier__min_samples_leaf=8, classifier__n_estimators=10),
        },
    )


# noinspection DuplicatedCode
def test_model_inspection_classifier_binary(
    iris_sample_binary: Sample, iris_classifier_crossfit_binary, n_jobs: int
) -> None:

    model_inspector = LearnerInspector(shap_interaction=False, n_jobs=n_jobs).fit(
        crossfit=iris_classifier_crossfit_binary
    )

    # calculate the shap value matrix, without any consolidation
    shap_values = model_inspector.shap_values(aggregation=None)

    # do the shap values add up to predictions minus a constant value?
    _validate_shap_values_against_predictions(
        shap_values=shap_values, crossfit=iris_classifier_crossfit_binary
    )

    shap_matrix_mean = model_inspector.shap_values()

    # is the consolidation correct?
    assert_frame_equal(shap_matrix_mean, shap_values.mean(level=1))

    # the length of rows in shap_values should be equal to the unique observation
    # indices we have had in the predictions_df
    assert len(shap_matrix_mean) == len(iris_sample_binary)

    # Shap decomposition matrices (feature dependencies)

    try:
        association_matrix = model_inspector.feature_association_matrix(
            clustered=True, symmetrical=True
        )
        assert association_matrix.values == pytest.approx(
            np.array(
                [
                    [1.000, 0.692, 0.195, 0.052],
                    [0.692, 1.000, 0.290, 0.041],
                    [0.195, 0.290, 1.000, 0.081],
                    [0.052, 0.041, 0.081, 1.000],
                ]
            ),
            abs=0.02,
        )
    except AssertionError as error:
        print_expected_matrix(error=error)
        raise

    linkage_tree = model_inspector.feature_association_linkage()

    print()
    DendrogramDrawer(style=DendrogramReportStyle()).draw(
        data=linkage_tree, title="Iris (binary) feature association linkage"
    )


def test_model_inspection_classifier_binary_single_shap_output() -> None:
    # simulate some data
    x, y = make_classification(
        n_samples=200, n_features=5, n_informative=5, n_redundant=0, random_state=42
    )
    sim_df = pd.DataFrame(
        np.hstack([x, y[:, np.newaxis]]),
        columns=[*(f"f{i}" for i in range(5)), "target"],
    )

    # create sample object
    sample_df = Sample(observations=sim_df, target_name="target")

    # fit the crossfit
    crossfit = LearnerCrossfit(
        pipeline=ClassifierPipelineDF(
            classifier=GradientBoostingClassifierDF(random_state=42)
        ),
        cv=BootstrapCV(n_splits=5, random_state=42),
        random_state=42,
        n_jobs=-3,
    ).fit(sample_df)

    # fit the inspector
    LearnerInspector(n_jobs=-3).fit(crossfit=crossfit)


# noinspection DuplicatedCode
def test_model_inspection_classifier_multi_class(
    iris_sample: Sample,
    iris_classifier_crossfit_multi_class: LearnerCrossfit[ClassifierPipelineDF],
    iris_inspector_multi_class: LearnerInspector[ClassifierPipelineDF],
    n_jobs: int,
) -> None:

    # calculate the shap value matrix, without any consolidation
    shap_values = iris_inspector_multi_class.shap_values(aggregation=None)

    # do the shap values add up to predictions minus a constant value?
    _validate_shap_values_against_predictions(
        shap_values=shap_values, crossfit=iris_classifier_crossfit_multi_class
    )

    shap_matrix_mean: List[pd.DataFrame] = iris_inspector_multi_class.shap_values()

    for _mean, _raw in zip(shap_matrix_mean, shap_values):
        # is the consolidation correct?
        assert_frame_equal(_mean, _raw.mean(level=1))

        # the length of rows in shap_values should be equal to the unique observation
        # indices we have had in the predictions_df
        assert len(_mean) == len(iris_sample)

    # Feature importance

    feature_importance: pd.DataFrame = iris_inspector_multi_class.feature_importance()
    assert feature_importance.index.equals(
        pd.Index(iris_sample.feature_names, name="feature")
    )
    assert feature_importance.columns.equals(
        pd.Index(iris_inspector_multi_class.output_names_, name="class")
    )
    assert feature_importance.values == pytest.approx(
        np.array(
            [
                [0.125, 0.085, 0.104],
                [0.020, 0.019, 0.010],
                [0.424, 0.456, 0.461],
                [0.432, 0.441, 0.425],
            ]
        ),
        abs=0.02,
    )

    # Shap decomposition matrices (feature dependencies)

    try:
        synergy_matrix = iris_inspector_multi_class.feature_synergy_matrix(
            clustered=False
        )

        assert np.hstack([m.values for m in synergy_matrix]) == pytest.approx(
            np.array(
                [
                    [1.000, 0.009, 0.057, 0.055, 1.000, 0.042]
                    + [0.418, 0.418, 1.000, 0.004, 0.085, 0.097],
                    [0.101, 1.000, 0.052, 0.072, 0.094, 1.000]
                    + [0.117, 0.156, 0.090, 1.000, 0.237, 0.258],
                    [0.003, 0.001, 1.000, 0.002, 0.027, 0.005]
                    + [1.000, 0.041, 0.012, 0.004, 1.000, 0.031],
                    [0.002, 0.000, 0.001, 1.000, 0.029, 0.005]
                    + [0.043, 1.000, 0.015, 0.005, 0.036, 1.000],
                ]
            ),
            abs=0.02,
        )

        redundancy_matrix = iris_inspector_multi_class.feature_redundancy_matrix(
            clustered=False
        )
        assert np.hstack([m.values for m in redundancy_matrix]) == (
            pytest.approx(
                np.array(
                    [
                        [1.000, 0.087, 0.643, 0.656, 1.000, 0.065]
                        + [0.265, 0.234, 1.000, 0.034, 0.594, 0.505],
                        [0.082, 1.000, 0.297, 0.292, 0.064, 1.000]
                        + [0.117, 0.171, 0.031, 1.000, 0.024, 0.021],
                        [0.682, 0.314, 1.000, 0.996, 0.471, 0.130]
                        + [1.000, 0.743, 0.642, 0.031, 1.000, 0.761],
                        [0.695, 0.315, 0.997, 1.000, 0.406, 0.194]
                        + [0.741, 1.000, 0.550, 0.028, 0.756, 1.000],
                    ]
                ),
                abs=0.02,
            )
        )

        association_matrix = iris_inspector_multi_class.feature_association_matrix(
            clustered=False
        )
        assert np.hstack([m.values for m in association_matrix]) == (
            pytest.approx(
                np.array(
                    [
                        [1.000, 0.077, 0.662, 0.670, 1.000, 0.046]
                        + [0.370, 0.334, 1.000, 0.031, 0.634, 0.550],
                        [0.077, 1.000, 0.301, 0.295, 0.046, 1.000]
                        + [0.127, 0.173, 0.031, 1.000, 0.025, 0.020],
                        [0.662, 0.301, 1.000, 0.998, 0.370, 0.127]
                        + [1.000, 0.783, 0.634, 0.025, 1.000, 0.790],
                        [0.670, 0.295, 0.998, 1.000, 0.334, 0.173]
                        + [0.783, 1.000, 0.550, 0.020, 0.790, 1.000],
                    ]
                ),
                abs=0.02,
            )
        )
    except AssertionError as error:
        print_expected_matrix(error=error, split=True)
        raise

    linkage_trees = iris_inspector_multi_class.feature_association_linkage()

    for output, linkage_tree in zip(
        iris_inspector_multi_class.output_names_, linkage_trees
    ):
        print()
        DendrogramDrawer(style=DendrogramReportStyle()).draw(
            data=linkage_tree, title=f"Iris feature association linkage: {output}"
        )


def _validate_shap_values_against_predictions(
    shap_values: pd.DataFrame, crossfit: LearnerCrossfit[ClassifierPipelineDF]
):

    # calculate the matching predictions, so we can check if the SHAP values add up
    # correctly
    predicted_probabilities_per_split: List[pd.DataFrame] = [
        model.predict_proba(crossfit.sample_.features.iloc[test_split, :])
        for model, (_, test_split) in zip(crossfit.models(), crossfit.splits())
    ]

    for split, predicted_probabilities in enumerate(predicted_probabilities_per_split):

        assert isinstance(
            predicted_probabilities, pd.DataFrame
        ), "predicted probabilities are single-output"

        expected_probability_range = 1 / len(predicted_probabilities.columns)

        def _check_probabilities(
            _class_probabilities: pd.DataFrame, _shap_for_split_and_class: pd.Series
        ) -> None:
            expected_probability = _class_probabilities.join(
                _shap_for_split_and_class
            ).sum(axis=1)

            expected_probability_min = expected_probability.min()
            expected_probability_max = expected_probability.max()
            assert expected_probability_min == pytest.approx(
                expected_probability_max
            ), "expected probability is the same for all explanations"
            assert (
                expected_probability_range * 0.6
                <= expected_probability_min
                <= expected_probability_range / 0.6
            ), (
                "expected class probability is roughly in the range of "
                f"{expected_probability_range * 100:.0f}%"
            )

        if predicted_probabilities.shape[1] == 2:
            # for binary classification we have SHAP values only for the second class
            _check_probabilities(
                predicted_probabilities.iloc[:, [1]],
                -shap_values.xs(split).sum(axis=1).rename("shap"),
            )

        else:
            # multi-class classification has outputs for each class

            for class_idx, class_name in enumerate(predicted_probabilities.columns):
                # for each observation and class, we expect to get the constant
                # expected probability value by deducting the SHAP values for all
                # features from the predicted probability

                class_probabilities = predicted_probabilities.loc[:, [class_name]]

                shap_for_split_and_class = (
                    -shap_values[class_idx].xs(split).sum(axis=1).rename("shap")
                )

                _check_probabilities(class_probabilities, shap_for_split_and_class)


# noinspection DuplicatedCode
def test_model_inspection_classifier_interaction(
    iris_sample_binary: Sample,
    iris_classifier_crossfit_binary: LearnerCrossfit[
        ClassifierPipelineDF[RandomForestClassifierDF]
    ],
    n_jobs: int,
) -> None:
    warnings.filterwarnings("ignore", message="You are accessing a training score")

    model_inspector = LearnerInspector(
        explainer_factory=TreeExplainerFactory(
            feature_perturbation="tree_path_dependent", use_background_dataset=True
        ),
        n_jobs=n_jobs,
    ).fit(crossfit=iris_classifier_crossfit_binary)

    model_inspector_full_sample = LearnerInspector(
        explainer_factory=TreeExplainerFactory(
            feature_perturbation="tree_path_dependent", use_background_dataset=True
        ),
        n_jobs=n_jobs,
    ).fit(crossfit=iris_classifier_crossfit_binary, full_sample=True)

    # disable legacy calculations; we used them in the constructor so the legacy
    # SHAP decomposer is created along with the new SHAP vector projector
    model_inspector._legacy = False

    model_inspector_no_interaction = LearnerInspector(
        shap_interaction=False,
        explainer_factory=TreeExplainerFactory(
            feature_perturbation="tree_path_dependent", use_background_dataset=True
        ),
        n_jobs=n_jobs,
    ).fit(crossfit=iris_classifier_crossfit_binary)

    # calculate shap interaction values
    shap_interaction_values = model_inspector.shap_interaction_values()

    # calculate shap values from interaction values
    shap_values = shap_interaction_values.groupby(by="observation").sum()

    # shap interaction values add up to shap values
    # we have to live with differences of up to 0.020, given the different results
    # returned for SHAP values and SHAP interaction values
    # todo: review accuracy after implementing use of a background dataset
    assert (
        model_inspector_no_interaction.shap_values() - shap_values
    ).abs().max().max() < 0.015

    # the column names of the shap value data frames are the feature names
    feature_columns = iris_sample_binary.feature_names
    assert shap_values.columns.to_list() == feature_columns
    assert shap_interaction_values.columns.to_list() == feature_columns

    # the length of rows in shap_values should be equal to the number of observations
    assert len(shap_values) == len(iris_sample_binary)

    # the length of rows in shap_interaction_values should be equal to the number of
    # observations, times the number of features
    assert len(shap_interaction_values) == (
        len(iris_sample_binary) * len(feature_columns)
    )

    # do the shap values add up to predictions minus a constant value?
    _validate_shap_values_against_predictions(
        shap_values=model_inspector.shap_interaction_values(aggregation=None)
        .groupby(level=[0, 1])
        .sum(),
        crossfit=iris_classifier_crossfit_binary,
    )

    assert model_inspector.feature_importance().values == pytest.approx(
        np.array([0.063, 0.013, 0.492, 0.431]), abs=0.02
    )

    try:
        synergy_matrix = model_inspector.feature_synergy_matrix(
            clustered=False, symmetrical=True
        )
        assert synergy_matrix.values == pytest.approx(
            np.array(
                [
                    [1.000, 0.011, 0.006, 0.007],
                    [0.011, 1.000, 0.006, 0.007],
                    [0.006, 0.006, 1.000, 0.003],
                    [0.007, 0.007, 0.003, 1.000],
                ]
            ),
            abs=0.02,
        )
        assert model_inspector.feature_synergy_matrix(
            absolute=True, symmetrical=True
        ).values == pytest.approx(
            np.array(
                [
                    [0.425, 0.001, 0.002, 0.001],
                    [0.001, 0.019, 0.000, 0.002],
                    [0.002, 0.000, 0.068, 0.002],
                    [0.001, 0.002, 0.002, 0.488],
                ]
            ),
            abs=0.02,
        )

        synergy_matrix = model_inspector.feature_synergy_matrix(clustered=True)
        assert synergy_matrix.values == pytest.approx(
            np.array(
                [
                    [1.000, 0.000, 0.001, 0.004],
                    [0.149, 1.000, 0.045, 0.157],
                    [0.040, 0.004, 1.000, 0.044],
                    [0.003, 0.001, 0.001, 1.000],
                ]
            ),
            abs=0.02,
        )
        assert model_inspector.feature_synergy_matrix(
            absolute=True
        ).values == pytest.approx(
            np.array(
                [
                    [0.425, 0.000, 0.000, 0.001],
                    [0.003, 0.019, 0.001, 0.003],
                    [0.003, 0.000, 0.068, 0.003],
                    [0.001, 0.000, 0.001, 0.488],
                ]
            ),
            abs=0.02,
        )
        assert model_inspector_full_sample.feature_synergy_matrix(
            clustered=True
        ).values == pytest.approx(
            np.array(
                [
                    [1.000, 0.000, 0.000, 0.001],
                    [0.386, 1.000, 0.108, 0.314],
                    [0.005, 0.002, 1.000, 0.059],
                    [0.002, 0.000, 0.001, 1.000],
                ]
            ),
            abs=0.02,
        )

        redundancy_matrix = model_inspector.feature_redundancy_matrix(
            clustered=False, symmetrical=True
        )
        assert redundancy_matrix.values == pytest.approx(
            np.array(
                [
                    [1.000, 0.080, 0.316, 0.208],
                    [0.080, 1.000, 0.036, 0.044],
                    [0.316, 0.036, 1.000, 0.691],
                    [0.208, 0.044, 0.691, 1.000],
                ]
            ),
            abs=0.02,
        )
        assert model_inspector.feature_redundancy_matrix(
            absolute=True, symmetrical=True
        ).values == pytest.approx(
            np.array(
                [
                    [0.425, 0.316, 0.052, 0.010],
                    [0.316, 0.488, 0.087, 0.009],
                    [0.052, 0.087, 0.068, 0.004],
                    [0.010, 0.009, 0.004, 0.019],
                ]
            ),
            abs=0.02,
        )

        redundancy_matrix = model_inspector.feature_redundancy_matrix(clustered=True)
        assert redundancy_matrix.values == pytest.approx(
            np.array(
                [
                    [1.000, 0.691, 0.209, 0.045],
                    [0.692, 1.000, 0.317, 0.037],
                    [0.201, 0.303, 1.000, 0.081],
                    [0.040, 0.031, 0.076, 1.000],
                ]
            ),
            abs=0.02,
        )
        assert model_inspector.feature_redundancy_matrix(
            absolute=True
        ).values == pytest.approx(
            np.array(
                [
                    [0.425, 0.294, 0.092, 0.020],
                    [0.337, 0.488, 0.154, 0.017],
                    [0.013, 0.020, 0.068, 0.006],
                    [0.001, 0.001, 0.001, 0.019],
                ]
            ),
            abs=0.02,
        )

        assert model_inspector_full_sample.feature_redundancy_matrix(
            clustered=True
        ).values == pytest.approx(
            np.array(
                [
                    [1.000, 0.677, 0.384, 0.003],
                    [0.676, 1.000, 0.465, 0.000],
                    [0.382, 0.438, 1.000, 0.013],
                    [0.002, 0.000, 0.012, 1.000],
                ]
            ),
            abs=0.02,
        )

        association_matrix = model_inspector.feature_association_matrix(
            clustered=False, symmetrical=True
        )
        assert association_matrix.values == pytest.approx(
            np.array(
                [
                    [1.000, 0.074, 0.309, 0.205],
                    [0.074, 1.000, 0.030, 0.040],
                    [0.309, 0.030, 1.000, 0.694],
                    [0.205, 0.040, 0.694, 1.000],
                ]
            ),
            abs=0.02,
        )
        assert model_inspector.feature_association_matrix(
            absolute=True, symmetrical=True
        ).values == pytest.approx(
            np.array(
                [
                    [0.425, 0.317, 0.051, 0.009],
                    [0.317, 0.488, 0.085, 0.007],
                    [0.051, 0.085, 0.068, 0.003],
                    [0.009, 0.007, 0.003, 0.019],
                ]
            ),
            abs=0.02,
        )

        association_matrix = model_inspector.feature_association_matrix(clustered=True)
        assert association_matrix.values == pytest.approx(
            np.array(
                [
                    [1.000, 0.694, 0.205, 0.040],
                    [0.694, 1.000, 0.309, 0.030],
                    [0.205, 0.309, 1.000, 0.074],
                    [0.040, 0.030, 0.074, 1.000],
                ]
            ),
            abs=0.02,
        )
        assert model_inspector.feature_association_matrix(
            absolute=True
        ).values == pytest.approx(
            np.array(
                [
                    [0.425, 0.295, 0.090, 0.018],
                    [0.338, 0.488, 0.150, 0.014],
                    [0.013, 0.020, 0.068, 0.005],
                    [0.001, 0.001, 0.001, 0.019],
                ]
            ),
            abs=0.02,
        )

        assert model_inspector_full_sample.feature_association_matrix(
            clustered=True
        ).values == pytest.approx(
            np.array(
                [
                    [1.000, 0.678, 0.383, 0.001],
                    [0.678, 1.000, 0.447, 0.000],
                    [0.383, 0.447, 1.000, 0.009],
                    [0.001, 0.000, 0.009, 1.000],
                ]
            ),
            abs=0.02,
        )

    except AssertionError as error:
        print_expected_matrix(error=error)
        raise

    linkage_tree = model_inspector.feature_redundancy_linkage()

    print()
    DendrogramDrawer(style=DendrogramReportStyle()).draw(
        data=linkage_tree, title="Iris (binary) feature redundancy linkage"
    )


def test_model_inspection_classifier_interaction_dual_target(
    iris_sample_binary_dual_target: Sample,
    iris_classifier_ranker_dual_target: LearnerRanker[
        ClassifierPipelineDF[RandomForestClassifierDF]
    ],
    iris_target_name,
    n_jobs: int,
) -> None:
    iris_classifier_crossfit_dual_target = (
        iris_classifier_ranker_dual_target.best_model_crossfit_
    )

    with pytest.raises(
        ValueError,
        match=(
            f"only single-output classifiers .* are supported.*"
            f"{iris_target_name}.*{iris_target_name}2"
        ),
    ):
        LearnerInspector(n_jobs=n_jobs).fit(
            crossfit=iris_classifier_crossfit_dual_target
        )


def test_shap_plot_data(
    iris_sample,
    iris_inspector_multi_class: LearnerInspector[ClassifierPipelineDF],
) -> None:
    shap_plot_data = iris_inspector_multi_class.shap_plot_data()
    # noinspection SpellCheckingInspection
    assert tuple(iris_inspector_multi_class.output_names_) == (
        "setosa",
        "versicolor",
        "virginica",
    )

    features_shape = shap_plot_data.features.shape
    shap_values = shap_plot_data.shap_values
    assert isinstance(shap_values, list)
    assert len(shap_values) == 3
    assert all(isinstance(shap, np.ndarray) for shap in shap_values)
    assert all(shap.shape == features_shape for shap in shap_values)

    shap_index = shap_plot_data.features.index
    assert_frame_equal(shap_plot_data.features, iris_sample.features.loc[shap_index])
    assert_series_equal(shap_plot_data.target, iris_sample.target.loc[shap_index])


#
# Utility functions
#


def fit_learner_ranker(
    sample: Sample, cv: BaseCrossValidator, n_jobs: int
) -> LearnerRanker[ClassifierPipelineDF[RandomForestClassifierDF]]:
    # define parameters and crossfit
    grids = [
        LearnerGrid(
            pipeline=ClassifierPipelineDF(
                classifier=RandomForestClassifierDF(random_state=42), preprocessing=None
            ),
            learner_parameters={"n_estimators": [10, 50], "min_samples_leaf": [4, 8]},
        )
    ]
    # pipeline inspector does only support binary classification - hence
    # filter the test_sample down to only 2 target classes:
    return LearnerRanker(
        grids=grids,
        cv=cv,
        scoring="f1_macro",
        random_state=42,
        n_jobs=n_jobs,
    ).fit(sample=sample)


def print_expected_matrix(error: AssertionError, split: bool = False):
    # used to print expected output for copy/paste into assertion statement

    import re

    matrix: List[List[float]] = eval(
        re.search(r"array\(([^)]+)\)", error.args[0])[1].replace(r"\n", "\n")
    )

    print("==== matrix assertion failed ====\nExpected Matrix:")
    print("[")
    for row in matrix:
        txt = "    ["
        halfpoint = len(row) // 2
        for i, x in enumerate(row):
            if split and i == halfpoint:
                txt += "] + ["
            elif i > 0:
                txt += ","
            txt += f"{x:.3f}"
        print(txt + "],")
    print("]")
