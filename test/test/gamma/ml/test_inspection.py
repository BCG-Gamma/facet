import logging
import warnings
from typing import *

import numpy as np
import pandas as pd
from pandas.core.util.hashing import hash_pandas_object
from shap import KernelExplainer, TreeExplainer
from shap.explainers.explainer import Explainer
from sklearn.base import BaseEstimator
from sklearn.model_selection import BaseCrossValidator, RepeatedKFold

from gamma.ml import Sample
from gamma.ml.fitcv import ClassifierFitCV, RegressorFitCV
from gamma.ml.inspection import ClassifierInspector, RegressorInspector
from gamma.ml.selection import (
    ModelEvaluation,
    ModelRanker,
    ParameterGrid,
    summary_report,
)
from gamma.ml.validation import CircularCV
from gamma.ml.viz import DendrogramDrawer, DendrogramReportStyle
from gamma.sklearndf import TransformerDF
from gamma.sklearndf.classification import RandomForestClassifierDF
from gamma.sklearndf.pipeline import ClassifierPipelineDF, RegressorPipelineDF
from gamma.sklearndf.regression import LGBMRegressorDF, SVRDF

log = logging.getLogger(__name__)

K_FOLDS: int = 5
TEST_RATIO = 1 / K_FOLDS
N_SPLITS = K_FOLDS * 2


def test_model_inspection(n_jobs, boston_sample: Sample) -> None:

    # define checksums for this test - one for the LGBM, one for the SVR
    CHKSUMS_PREDICTIONS = (8498474725463556484, 781552878134992853)
    CHKSUMS_SHAP = (5120923735415774388, 2945880188040048636)
    CHKSUM_CORR_MATRIX = (4159152513108370414, 16061019524360971856)

    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
    warnings.filterwarnings("ignore", message="You are accessing a training score")

    # define a CV:
    # noinspection PyTypeChecker
    test_cv: BaseCrossValidator = RepeatedKFold(
        n_splits=K_FOLDS, n_repeats=N_SPLITS // K_FOLDS, random_state=42
    )

    # define parameters and models
    models = [
        ParameterGrid(
            pipeline=(
                RegressorPipelineDF(regressor=SVRDF(gamma="scale"), preprocessing=None)
            ),
            estimator_parameters={"kernel": ("linear", "rbf"), "C": [1, 10]},
        ),
        ParameterGrid(
            pipeline=RegressorPipelineDF(
                regressor=LGBMRegressorDF(), preprocessing=None
            ),
            estimator_parameters={
                "max_depth": (1, 2, 5),
                "min_split_gain": (0.1, 0.2, 0.5),
                "num_leaves": (2, 3),
            },
        ),
    ]

    # use first 100 rows only, since KernelExplainer is very slow...

    test_sample: Sample = boston_sample.select_observations_by_position(
        positions=range(0, 100)
    )

    model_ranker = ModelRanker(
        grids=models, cv=test_cv, scoring="neg_mean_squared_error"
    )

    model_ranking = model_ranker.run(test_sample, n_jobs=n_jobs)

    log.debug(f"\n{summary_report(model_ranking[:10])}")

    # consider: model_with_type(...) function for ModelRanking
    best_svr = [m for m in model_ranking if isinstance(m.model.regressor, SVRDF)][0]
    best_lgbm = [
        model_evaluation
        for model_evaluation in model_ranking
        if isinstance(model_evaluation.model.regressor, LGBMRegressorDF)
    ][0]

    for model_index, model_evaluation in enumerate((best_lgbm, best_svr)):

        model_fit = RegressorFitCV(
            pipeline=model_evaluation.model, cv=test_cv, sample=test_sample
        )

        # test predictions_for_all_samples
        predictions_df = model_fit.predictions_for_all_splits()
        assert RegressorFitCV.F_PREDICTION in predictions_df.columns
        assert RegressorFitCV.F_TARGET in predictions_df.columns

        # check number of split ids
        assert (
            predictions_df.index.get_level_values(
                level=RegressorFitCV.F_SPLIT_ID
            ).nunique()
            == N_SPLITS
        )

        # check correct number of rows
        allowed_variance = 0.01
        assert (
            (len(test_sample) * (TEST_RATIO - allowed_variance) * N_SPLITS)
            <= len(predictions_df)
            <= (len(test_sample) * (TEST_RATIO + allowed_variance) * N_SPLITS)
        )

        # check actual prediction values using checksum:
        assert (
            np.sum(hash_pandas_object(predictions_df).values)
            == CHKSUMS_PREDICTIONS[model_index]
        )

        model_inspector = RegressorInspector(models=model_fit)
        # make and check shap value matrix
        shap_matrix = model_inspector.shap_matrix()

        # the length of rows in shap_matrix should be equal to the unique observation
        # indices we have had in the predictions_df
        assert len(shap_matrix) == len(test_sample)

        # check actual values using checksum:
        #
        assert (
            np.sum(hash_pandas_object(shap_matrix.round(decimals=4)).values)
            == CHKSUMS_SHAP[model_index]
        )

        # correlated shap matrix: feature dependencies
        corr_matrix: pd.DataFrame = model_inspector.feature_dependency_matrix()

        # check number of rows
        assert len(corr_matrix) == len(test_sample.features.columns) - 1
        assert len(corr_matrix.columns) == len(test_sample.features.columns) - 1

        # check correlation values
        for c in corr_matrix.columns:
            assert (
                -1.0
                <= corr_matrix.fillna(0).loc[:, c].min()
                <= corr_matrix.fillna(0).loc[:, c].max()
                <= 1.0
            )

        # check actual values using checksum:
        assert (
            np.sum(hash_pandas_object(corr_matrix.round(decimals=4)).values)
            == CHKSUM_CORR_MATRIX[model_index]
        )

        linkage_tree = model_inspector.cluster_dependent_features()

        DendrogramDrawer(
            title="Test", linkage_tree=linkage_tree, style=DendrogramReportStyle()
        ).draw()


def test_model_inspection_with_encoding(
    batch_table: pd.DataFrame,
    regressor_grids: Sequence[ParameterGrid],
    sample: Sample,
    simple_preprocessor: TransformerDF,
    n_jobs,
) -> None:

    # define checksums for this test
    CHKSUM_PREDICTIONS = 15445317532513327074
    CHKSUM_SHAP = 2454983946504277938
    CHKSUM_CORR_MATRIX = 9841870561220906358

    # define the circular cross validator with just 5 splits (to speed up testing)
    circular_cv = CircularCV(test_ratio=0.20, n_splits=5)

    model_ranker: ModelRanker = ModelRanker(
        grids=regressor_grids, cv=circular_cv, scoring="r2"
    )

    # run the ModelRanker to retrieve a ranking
    model_ranking: Sequence[ModelEvaluation] = model_ranker.run(
        sample=sample, n_jobs=n_jobs
    )

    log.debug(f"\n{summary_report(model_ranking[:10])}")

    # we get the best model_evaluation which is a LGBM - for the sake of test
    # performance
    model_evaluation = [
        model_evaluation
        for model_evaluation in model_ranking
        if isinstance(model_evaluation.model.regressor, LGBMRegressorDF)
    ][0]

    model_fit = RegressorFitCV(
        pipeline=model_evaluation.model, cv=circular_cv, sample=sample
    )

    predictions = model_fit.predictions_for_all_splits()

    # check actual values using checksum:
    assert (
        np.sum(hash_pandas_object(predictions.round(decimals=4)).values)
        == CHKSUM_PREDICTIONS
    )

    mi = RegressorInspector(models=model_fit)

    shap_matrix = mi.shap_matrix()

    # check actual values using checksum:
    assert (
        np.sum(hash_pandas_object(shap_matrix.round(decimals=4)).values) == CHKSUM_SHAP
    )

    # correlated shap matrix: feature dependencies
    corr_matrix: pd.DataFrame = mi.feature_dependency_matrix()

    # check actual values using checksum:
    assert (
        np.sum(hash_pandas_object(corr_matrix.round(decimals=4)).values)
        == CHKSUM_CORR_MATRIX
    )

    # cluster feature importances
    linkage_tree = mi.cluster_dependent_features()

    #  test the ModelInspector with a custom ExplainerFactory:
    def ef(estimator: BaseEstimator, data: pd.DataFrame) -> Explainer:

        try:
            return TreeExplainer(
                model=estimator, feature_dependence="independent", data=data
            )
        except Exception as e:
            log.debug(
                f"failed to instantiate shap.TreeExplainer:{str(e)},"
                "using shap.KernelExplainer as fallback"
            )
            # noinspection PyUnresolvedReferences
            return KernelExplainer(model=estimator.predict, data=data)

    mi2 = RegressorInspector(models=model_fit, explainer_factory=ef)
    mi2.shap_matrix()

    linkage_tree = mi2.cluster_dependent_features()
    print()
    DendrogramDrawer(
        title="Test", linkage_tree=linkage_tree, style=DendrogramReportStyle()
    ).draw()


def test_model_inspection_classifier(n_jobs, iris_sample: Sample) -> None:
    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
    warnings.filterwarnings("ignore", message="You are accessing a training score")

    # define checksums for this test
    CHKSUM_PREDICTIONS = 13787062061417505156
    CHKSUM_SHAP = 12636830693175052845
    CHKSUM_CORR_MATRIX = 9331449050691600977

    # define a CV:
    # noinspection PyTypeChecker
    test_cv: BaseCrossValidator = RepeatedKFold(
        n_splits=K_FOLDS, n_repeats=N_SPLITS // K_FOLDS, random_state=42
    )

    # define parameters and models
    models = [
        ParameterGrid(
            pipeline=ClassifierPipelineDF(
                classifier=RandomForestClassifierDF(), preprocessing=None
            ),
            estimator_parameters={"n_estimators": [50, 80], "random_state": [42]},
        )
    ]

    # model inspector does only support binary classification - hence
    # filter the test_sample down to only 2 target classes:
    test_sample: Sample = iris_sample.select_observations_by_index(
        ids=iris_sample.target.isin(iris_sample.target.unique()[0:2])
    )

    model_ranker: ModelRanker = ModelRanker(
        grids=models, cv=test_cv, scoring="f1_macro"
    )

    model_ranking: Sequence[ModelEvaluation] = model_ranker.run(
        test_sample, n_jobs=n_jobs
    )

    log.debug(f"\n{summary_report(model_ranking[:10])}")

    # consider: model_with_type(...) function for ModelRanking
    model_evaluation = model_ranking[0]

    model_fit = ClassifierFitCV(
        pipeline=model_evaluation.model,
        cv=test_cv,
        sample=test_sample,
        calibration=ClassifierFitCV.SIGMOID,
        n_jobs=n_jobs,
    )

    predictions = model_fit.predictions_for_all_splits()

    # check actual values using checksum:
    assert (
        np.sum(hash_pandas_object(predictions.round(decimals=4)).values)
        == CHKSUM_PREDICTIONS
    )

    model_inspector = ClassifierInspector(models=model_fit)
    # make and check shap value matrix
    shap_matrix = model_inspector.shap_matrix()

    # check actual values using checksum:
    assert (
        np.sum(hash_pandas_object(shap_matrix.round(decimals=4)).values) == CHKSUM_SHAP
    )

    # the length of rows in shap_matrix should be equal to the unique observation
    # indices we have had in the predictions_df
    assert len(shap_matrix) == len(test_sample)

    # correlated shap matrix: feature dependencies
    corr_matrix: pd.DataFrame = model_inspector.feature_dependency_matrix()
    log.info(corr_matrix)
    # check number of rows
    assert len(corr_matrix) == len(test_sample.features.columns)
    assert len(corr_matrix.columns) == len(test_sample.features.columns)

    # check correlation values
    for c in corr_matrix.columns:
        assert (
            -1.0
            <= corr_matrix.fillna(0).loc[:, c].min()
            <= corr_matrix.fillna(0).loc[:, c].max()
            <= 1.0
        )

    # check actual values using checksum:
    assert (
        np.sum(hash_pandas_object(corr_matrix.round(decimals=4)).values)
        == CHKSUM_CORR_MATRIX
    )

    linkage_tree = model_inspector.cluster_dependent_features()
    print()
    DendrogramDrawer(
        title="Test", linkage_tree=linkage_tree, style=DendrogramReportStyle()
    ).draw()
