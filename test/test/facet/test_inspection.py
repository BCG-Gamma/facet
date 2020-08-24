"""
Model inspector tests.
"""
import logging
import warnings
from typing import *

import numpy as np
import pandas as pd
import pytest
from pandas.util.testing import assert_frame_equal
from sklearn.model_selection import BaseCrossValidator, KFold

from facet import Sample
from facet.crossfit import LearnerCrossfit
from facet.inspection import KernelExplainerFactory, LearnerInspector
from facet.selection import LearnerGrid, LearnerRanker
from facet.validation import BootstrapCV, StratifiedBootstrapCV
from pytools.viz.dendrogram import DendrogramDrawer, DendrogramReportStyle
from sklearndf import TransformerDF
from sklearndf.classification import RandomForestClassifierDF
from sklearndf.pipeline import ClassifierPipelineDF, RegressorPipelineDF
from . import check_ranking

# noinspection PyMissingOrEmptyDocstring

log = logging.getLogger(__name__)


@pytest.fixture
def iris_classifier_ranker_binary(
    iris_sample_binary: Sample,
    cv_stratified_bootstrap: StratifiedBootstrapCV,
    n_jobs: int,
) -> LearnerRanker[ClassifierPipelineDF[RandomForestClassifierDF]]:
    return _fit_learner_ranker(
        sample=iris_sample_binary, cv=cv_stratified_bootstrap, n_jobs=n_jobs
    )


@pytest.fixture
def iris_classifier_ranker_multi_class(
    iris_sample: Sample, cv_stratified_bootstrap: StratifiedBootstrapCV, n_jobs: int
) -> LearnerRanker[ClassifierPipelineDF[RandomForestClassifierDF]]:
    return _fit_learner_ranker(
        sample=iris_sample, cv=cv_stratified_bootstrap, n_jobs=n_jobs
    )


@pytest.fixture
def iris_classifier_ranker_dual_target(
    iris_sample_binary_dual_target: Sample, cv_bootstrap: BootstrapCV, n_jobs: int
) -> LearnerRanker[ClassifierPipelineDF[RandomForestClassifierDF]]:
    return _fit_learner_ranker(
        sample=iris_sample_binary_dual_target, cv=cv_bootstrap, n_jobs=n_jobs
    )


@pytest.fixture
def iris_classifier_crossfit_binary(
    iris_classifier_ranker_binary: LearnerRanker[ClassifierPipelineDF]
) -> LearnerCrossfit[ClassifierPipelineDF[RandomForestClassifierDF]]:
    return iris_classifier_ranker_binary.best_model_crossfit


@pytest.fixture
def iris_classifier_crossfit_multi_class(
    iris_classifier_ranker_multi_class: LearnerRanker[ClassifierPipelineDF]
) -> LearnerCrossfit[ClassifierPipelineDF[RandomForestClassifierDF]]:
    return iris_classifier_ranker_multi_class.best_model_crossfit


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
    expected_scores = [0.418, 0.4, 0.386, 0.385, 0.122] + [
        0.122,
        -0.074,
        -0.074,
        -0.074,
        -0.074,
    ]

    log.debug(f"\n{regressor_ranker.summary_report(max_learners=10)}")

    check_ranking(
        ranking=regressor_ranker.ranking(),
        expected_scores=expected_scores,
        expected_learners=None,
        expected_parameters=None,
    )

    # using an invalid consolidation method raises an exception
    with pytest.raises(ValueError, match="unknown consolidation method: invalid"):
        regressor_inspector.shap_values(consolidate="invalid")

    shap_values_raw = regressor_inspector.shap_values(consolidate=None)
    shap_values_mean = regressor_inspector.shap_values(consolidate="mean")
    shap_values_std = regressor_inspector.shap_values(consolidate="std")

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
    assert shap_values_raw.index.names == (["split", "observation"])
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

    log.debug(f"\n{iris_classifier_ranker_binary.summary_report(max_learners=10)}")
    check_ranking(
        ranking=iris_classifier_ranker_binary.ranking(),
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
    shap_values = model_inspector.shap_values(consolidate=None)

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

    assert model_inspector.feature_association_matrix().values == pytest.approx(
        np.array(
            [
                [1.0, 0.028, 0.14, 0.128],
                [0.028, 1.0, 0.005, 0.002],
                [0.14, 0.005, 1.0, 0.681],
                [0.128, 0.002, 0.681, 1.0],
            ]
        ),
        abs=0.02,
    )

    linkage_tree = model_inspector.feature_association_linkage()

    print()
    DendrogramDrawer(style=DendrogramReportStyle()).draw(
        data=linkage_tree, title="Iris (binary) feature association linkage"
    )


# noinspection DuplicatedCode
def test_model_inspection_classifier_multi_class(
    iris_sample: Sample,
    iris_classifier_crossfit_multi_class: LearnerCrossfit[ClassifierPipelineDF],
    n_jobs: int,
) -> None:

    model_inspector = LearnerInspector(shap_interaction=True, n_jobs=n_jobs).fit(
        crossfit=iris_classifier_crossfit_multi_class
    )

    # calculate the shap value matrix, without any consolidation
    shap_values = model_inspector.shap_values(consolidate=None)

    # do the shap values add up to predictions minus a constant value?
    _validate_shap_values_against_predictions(
        shap_values=shap_values, crossfit=iris_classifier_crossfit_multi_class
    )

    shap_matrix_mean = model_inspector.shap_values()

    # is the consolidation correct?
    assert_frame_equal(shap_matrix_mean, shap_values.mean(level=1))

    # the length of rows in shap_values should be equal to the unique observation
    # indices we have had in the predictions_df
    assert len(shap_matrix_mean) == len(iris_sample)

    # Shap decomposition matrices (feature dependencies)

    assert model_inspector.feature_synergy_matrix().values == pytest.approx(
        np.array(
            [
                [1.0, 0.069, 0.081, 0.061, 1.0, 0.098]
                + [0.13, 0.118, 1.0, 0.022, 0.101, 0.101],
                [0.069, 1.0, 0.011, 0.012, 0.098, 1.0]
                + [0.019, 0.022, 0.022, 1.0, 0.016, 0.018],
                [0.081, 0.011, 1.0, 0.023, 0.13, 0.019]
                + [1.0, 0.205, 0.101, 0.016, 1.0, 0.165],
                [0.061, 0.012, 0.023, 1.0, 0.118, 0.022]
                + [0.205, 1.0, 0.101, 0.018, 0.165, 1.0],
            ]
        ),
        abs=0.02,
    )
    assert model_inspector.feature_redundancy_matrix().values == pytest.approx(
        np.array(
            [
                [1.0, 0.145, 0.418, 0.402, 1.0, 0.163]
                + [0.149, 0.144, 1.0, 0.011, 0.322, 0.326],
                [0.145, 1.0, 0.056, 0.053, 0.163, 1.0]
                + [0.034, 0.033, 0.011, 1.0, 0.0, 0.004],
                [0.418, 0.056, 1.0, 0.968, 0.149, 0.034]
                + [1.0, 0.63, 0.322, 0.0, 1.0, 0.803],
                [0.402, 0.053, 0.968, 1.0, 0.144, 0.033]
                + [0.63, 1.0, 0.326, 0.004, 0.803, 1.0],
            ]
        ),
        abs=0.02,
    )
    assert model_inspector.feature_association_matrix().values == pytest.approx(
        np.array(
            [
                [1.0, 0.098, 0.375, 0.373, 1.0, 0.109]
                + [0.207, 0.205, 1.0, 0.013, 0.302, 0.301],
                [0.098, 1.0, 0.056, 0.055, 0.109, 1.0]
                + [0.038, 0.044, 0.013, 1.0, 0.0, 0.0],
                [0.375, 0.056, 1.0, 0.982, 0.207, 0.038]
                + [1.0, 0.764, 0.302, 0.0, 1.0, 0.755],
                [0.373, 0.055, 0.982, 1.0, 0.205, 0.044]
                + [0.764, 1.0, 0.301, 0.0, 0.755, 1.0],
            ]
        ),
        abs=0.02,
    )

    linkage_trees = model_inspector.feature_association_linkage()

    for output, linkage_tree in zip(model_inspector.outputs, linkage_trees):
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
        model.predict_proba(crossfit.training_sample.features.iloc[test_split, :])
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
            # for binary classification we have SHAP values only for the first class
            _check_probabilities(
                predicted_probabilities.iloc[:, [0]],
                -shap_values.xs(split).sum(axis=1).rename("shap"),
            )

        else:
            # multi-class classification has outputs for each class

            for class_name in predicted_probabilities.columns:
                # for each observation and class, we expect to get the constant
                # expected probability value by deducting the SHAP values for all
                # features from the predicted probability

                class_probabilities = predicted_probabilities.loc[:, [class_name]]

                shap_for_split_and_class = (
                    -shap_values.xs(split)
                    .xs(class_name, axis=1)
                    .sum(axis=1)
                    .rename("shap")
                )

                _check_probabilities(class_probabilities, shap_for_split_and_class)


# noinspection DuplicatedCode
def test_model_inspection_classifier_interaction(
    iris_sample_binary: Sample, iris_classifier_crossfit_binary, n_jobs: int
) -> None:
    warnings.filterwarnings("ignore", message="You are accessing a training score")

    model_inspector = LearnerInspector(n_jobs=n_jobs).fit(
        crossfit=iris_classifier_crossfit_binary
    )

    model_inspector_no_interaction = LearnerInspector(
        shap_interaction=False, n_jobs=n_jobs
    ).fit(crossfit=iris_classifier_crossfit_binary)

    # calculate shap interaction values
    shap_interaction_values = model_inspector.shap_interaction_values()

    # calculate shap values from interaction values
    shap_values = shap_interaction_values.groupby(by="observation").sum()

    # shap interaction values add up to shap values
    # we have to live with differences of up to 0.02, given the different results
    # returned for SHAP values and SHAP interaction values
    # todo: review accuracy after implementing use of a background dataset
    assert (
        model_inspector_no_interaction.shap_values() - shap_values
    ).abs().max().max() < 0.015

    # the column names of the shap value data frames are the feature names
    feature_columns = iris_sample_binary.feature_columns
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
        shap_values=model_inspector.shap_interaction_values(consolidate=None)
        .groupby(level=[0, 1])
        .sum(),
        crossfit=iris_classifier_crossfit_binary,
    )

    assert model_inspector.feature_synergy_matrix().values == pytest.approx(
        np.array(
            [
                [1.0, 0.047, 0.101, 0.12],
                [0.047, 1.0, 0.017, 0.021],
                [0.101, 0.017, 1.0, 0.1],
                [0.12, 0.021, 0.1, 1.0],
            ]
        ),
        abs=0.02,
    )

    assert model_inspector.feature_redundancy_matrix().values == pytest.approx(
        np.array(
            [
                [1.0, 0.039, 0.181, 0.206],
                [0.039, 1.0, 0.005, 0.011],
                [0.181, 0.005, 1.0, 0.792],
                [0.206, 0.011, 0.792, 1.0],
            ]
        ),
        abs=0.02,
    )

    assert model_inspector.feature_association_matrix().values == pytest.approx(
        np.array(
            [
                [1.0, 0.028, 0.14, 0.128],
                [0.028, 1.0, 0.005, 0.002],
                [0.14, 0.005, 1.0, 0.681],
                [0.128, 0.002, 0.681, 1.0],
            ]
        ),
        abs=0.02,
    )

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
    n_jobs: int,
) -> None:
    iris_classifier_crossfit_dual_target = (
        iris_classifier_ranker_dual_target.best_model_crossfit
    )

    with pytest.raises(
        ValueError,
        match="only single-output classifiers .* are supported.*target.*target2",
    ):
        LearnerInspector(n_jobs=n_jobs).fit(
            crossfit=iris_classifier_crossfit_dual_target
        )


def _fit_learner_ranker(
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
        # shuffle_features=True,
        random_state=42,
        n_jobs=n_jobs,
    ).fit(sample=sample)
