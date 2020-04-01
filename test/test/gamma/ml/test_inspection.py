"""
Model inspector tests.
"""
import logging
import warnings
from typing import *

import numpy as np
import pandas as pd
from pandas.core.util.hashing import hash_pandas_object
from shap import KernelExplainer, TreeExplainer
from shap.explainers.explainer import Explainer
from sklearn.base import BaseEstimator
from sklearn.model_selection import BaseCrossValidator

from gamma.ml import Sample
from gamma.ml.crossfit import LearnerCrossfit
from gamma.ml.inspection import ClassifierInspector, RegressorInspector
from gamma.ml.selection import ClassifierRanker, ParameterGrid, RegressorRanker
from gamma.sklearndf import TransformerDF
from gamma.sklearndf.classification import RandomForestClassifierDF
from gamma.sklearndf.pipeline import ClassifierPipelineDF, RegressorPipelineDF
from gamma.viz.dendrogram import DendrogramDrawer, DendrogramReportStyle
from test.gamma.ml import check_ranking

log = logging.getLogger(__name__)


# noinspection PyMissingOrEmptyDocstring


def test_model_inspection(
    regressor_grids: Sequence[ParameterGrid],
    regressor_ranker: RegressorRanker,
    best_lgbm_crossfit: LearnerCrossfit[RegressorPipelineDF],
    regressor_inspector: RegressorInspector,
    cv: BaseCrossValidator,
    sample: Sample,
    simple_preprocessor: TransformerDF,
    n_jobs: int,
    fast_execution: bool,
) -> None:
    # define checksums for this test
    if fast_execution:
        checksum_shap = 7678718855667032507
        checksum_learner_scores = 1.5365912783588438
        checksum_learner_ranks = "ac87a8cbf8b279746707a2af8b66a7ac"
    else:
        checksum_shap = 1956741545033811954
        checksum_learner_scores = 0.6056819340325851
        checksum_learner_ranks = "4251e104ce7d1834f2b3b6ab5bb5ceab"

    log.debug(f"\n{regressor_ranker.summary_report(max_learners=10)}")

    check_ranking(
        ranking=regressor_ranker.ranking(),
        checksum_scores=checksum_learner_scores,
        checksum_learners=checksum_learner_ranks,
        first_n_learners=10,
    )

    shap_values = regressor_inspector.shap_values()

    # the length of rows in shap_values should be equal to the unique observation
    # indices we have had in the predictions_df
    assert len(shap_values) == len(sample)

    # check actual values using checksum:
    assert (
        np.sum(hash_pandas_object(shap_values.round(decimals=4)).values)
        == checksum_shap
    )

    #  test the ModelInspector with a custom ExplainerFactory:
    def _ef(estimator: BaseEstimator, data: pd.DataFrame) -> Explainer:

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

    # noinspection PyTypeChecker
    inspector_2 = RegressorInspector(explainer_factory=_ef, shap_interaction=False).fit(
        crossfit=best_lgbm_crossfit
    )
    inspector_2.shap_values()

    linkage_tree = inspector_2.feature_association_linkage()

    print()
    DendrogramDrawer(style="text").draw(data=linkage_tree, title="Test")


def test_model_inspection_classifier(
    iris_sample: Sample, cv: BaseCrossValidator, n_jobs: int
) -> None:
    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
    warnings.filterwarnings("ignore", message="You are accessing a training score")

    # define checksums for this test
    checksum_shap = 5207601201651574496
    checksum_association_matrix = 5535519327633455357
    checksum_learner_scores = 2.0
    checksum_learner_ranks = "a8fe61f0f98c078fbcf427ad344c1749"

    # define parameters and crossfit
    models = [
        ParameterGrid(
            pipeline=ClassifierPipelineDF(
                classifier=RandomForestClassifierDF(), preprocessing=None
            ),
            learner_parameters={"n_estimators": [50, 80], "random_state": [42]},
        )
    ]

    # pipeline inspector does only support binary classification - hence
    # filter the test_sample down to only 2 target classes:
    test_sample: Sample = iris_sample.subsample(
        loc=iris_sample.target.isin(iris_sample.target.unique()[0:2])
    )

    model_ranker = ClassifierRanker(
        grid=models,
        cv=cv,
        scoring="f1_macro",
        shuffle_features=True,
        random_state=42,
        n_jobs=n_jobs,
    ).fit(sample=test_sample)

    log.debug(f"\n{model_ranker.summary_report(max_learners=10)}")

    check_ranking(
        ranking=model_ranker.ranking(),
        checksum_scores=checksum_learner_scores,
        checksum_learners=checksum_learner_ranks,
        first_n_learners=10,
    )

    crossfit = model_ranker.best_model_crossfit

    model_inspector = ClassifierInspector(shap_interaction=False).fit(crossfit=crossfit)
    # make and check shap value matrix
    shap_matrix = model_inspector.shap_values()

    # check actual values using checksum:
    assert (
        np.sum(hash_pandas_object(shap_matrix.round(decimals=4)).values)
        == checksum_shap
    )

    # the length of rows in shap_values should be equal to the unique observation
    # indices we have had in the predictions_df
    assert len(shap_matrix) == len(test_sample)

    # Shap decomposition matrices (feature dependencies)
    feature_associations: pd.DataFrame = model_inspector.feature_association_matrix()
    log.info(feature_associations)
    # check number of rows
    assert len(feature_associations) == len(test_sample.feature_columns)
    assert len(feature_associations.columns) == len(test_sample.feature_columns)

    # check association values
    for c in feature_associations.columns:
        fa = feature_associations.loc[:, c]
        assert 0.0 <= fa.min() <= fa.max() <= 1.0

    # check actual values using checksum:
    assert (
        np.sum(hash_pandas_object(feature_associations.round(decimals=4)).values)
        == checksum_association_matrix
    )

    linkage_tree = model_inspector.feature_association_linkage()

    print()
    DendrogramDrawer(style=DendrogramReportStyle()).draw(
        data=linkage_tree, title="Test"
    )
