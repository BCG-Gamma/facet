#
# NOT FOR CLIENT USE!
#
# This is a pre-release library under development. Handling of IP rights is still
# being investigated. To avoid causing any potential IP disputes or issues, DO NOT USE
# ANY OF THIS CODE ON A CLIENT PROJECT, not even in modified form.
#
# Please direct any queries to any of:
# - Jan Ittner
# - Jörg Schneider
# - Florent Martin
#

"""
Inspection of a pipeline.

The :class:`ModelInspector` class computes the shap matrix and the associated linkage
tree of a pipeline which has been fitted using cross-validation.
"""
import logging
from abc import ABC, abstractmethod
from typing import *

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from shap import KernelExplainer, TreeExplainer
from shap.explainers.explainer import Explainer
from sklearn.base import BaseEstimator

from gamma.common import ListLike
from gamma.ml.crossfit import ClassifierCrossfit, LearnerCrossfit, RegressorCrossfit
from gamma.sklearndf.pipeline import (
    ClassifierPipelineDF,
    LearnerPipelineDF,
    RegressorPipelineDF,
)
from gamma.viz.dendrogram import LinkageTree

log = logging.getLogger(__name__)

__all__ = ["BaseLearnerInspector", "ClassifierInspector", "RegressorInspector"]


#
# Type variables
#

_T_LearnerPipelineDF = TypeVar("T_LearnerPipelineDF", bound=LearnerPipelineDF)
_T_RegressorPipelineDF = TypeVar("_T_RegressorPipelineDF", bound=RegressorPipelineDF)
_T_ClassifierPipelineDF = TypeVar("_T_ClassifierPipelineDF", bound=ClassifierPipelineDF)


#
# Class definitions
#


class BaseLearnerInspector(Generic[_T_LearnerPipelineDF], ABC):
    """
    Inspect a pipeline through its SHAP values.

    :param crossfit: predictor containing the information about the
      pipeline, the data (a Sample object), the cross-validation and crossfit.
    :param explainer_factory: calibration that returns a shap Explainer
    """

    __slots__ = [
        "_cross_fit",
        "_shap_matrix",
        "_feature_dependency_matrix",
        "_explainer_factory",
    ]

    COL_FEATURE = "feature"

    def __init__(
        self,
        crossfit: LearnerCrossfit[_T_LearnerPipelineDF],
        explainer_factory: Optional[
            Callable[[BaseEstimator, pd.DataFrame], Explainer]
        ] = None,
    ) -> None:
        if not crossfit.is_fitted:
            raise ValueError("arg crossfit expected to be fitted")

        self._cross_fit = crossfit
        self._shap_matrix: Optional[pd.DataFrame] = None
        self._feature_dependency_matrix: Optional[pd.DataFrame] = None
        self._explainer_factory = (
            explainer_factory
            if explainer_factory is not None
            else tree_explainer_factory
        )

    @property
    def crossfit(self) -> LearnerCrossfit[_T_LearnerPipelineDF]:
        """
        CV fit of the pipeline being examined by this inspector
        """
        return self._cross_fit

    def shap_matrix(self) -> pd.DataFrame:
        """
        Calculate the SHAP matrix for all splits.

        Each row is an observation in a specific test split, and each column is a
        feature, and values are the SHAP values per observation/split and feature.

        :return: shap matrix as a data frame
        """
        if self._shap_matrix is not None:
            return self._shap_matrix

        crossfit = self.crossfit

        shap_values_df = pd.concat(
            objs=[
                self._shap_matrix_for_split(model, test_indices)
                for model, (_, test_indices) in zip(
                    crossfit.models(), crossfit.splits()
                )
            ],
            sort=True,
        ).fillna(0.0)

        # Group SHAP matrix by observation ID and aggregate SHAP values using mean()
        self._shap_matrix = shap_values_df.groupby(by=shap_values_df.index).mean()

        return self._shap_matrix

    def _shap_matrix_for_split(
        self, split_model: LearnerPipelineDF, oob_indices: ListLike[int]
    ) -> pd.DataFrame:
        """
        Calculate the SHAP matrix for a single split.

        :param split_model: pipeline trained on the split
        :return: SHAP matrix of a single split as data frame
        """
        x_oob = self.crossfit.training_sample.select_observations_by_index(
            ids=oob_indices
        ).features

        estimator = split_model.final_estimator

        if split_model.preprocessing is not None:
            data_transformed = split_model.preprocessing.transform(x_oob)
        else:
            data_transformed = x_oob

        raw_shap_values = self._explainer_factory(
            estimator=estimator.root_estimator, data=data_transformed
        ).shap_values(data_transformed)

        return self._shap_matrix_for_split_to_df(
            raw_shap_values=raw_shap_values, split_transformed=data_transformed
        )

    @abstractmethod
    def _shap_matrix_for_split_to_df(
        self,
        raw_shap_values: Union[np.ndarray, List[np.ndarray]],
        split_transformed: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Convert the SHAP matrix for a single split to a data frame.

        :param raw_shap_values: the raw values returned by the SHAP explainer
        :param split_transformed: the transformed data the pipeline was trained on
        :return: SHAP matrix of a single split as data frame
        """
        pass

    def feature_importances(self) -> pd.Series:
        """
        Feature importance computed using absolute value of shap values.

        :return: feature importances as their mean absolute SHAP contributions,
          normalised to a total 100%
        """
        feature_importances: pd.Series = self.shap_matrix().abs().mean()
        return (feature_importances / feature_importances.sum()).sort_values(
            ascending=False
        )

    def feature_dependency_matrix(self) -> pd.DataFrame:
        """
        Return the Pearson correlation matrix of the shap matrix.

        :return: data frame with column and index given by the feature names,
          and values are the Pearson correlations of the shap values of features
        """
        if self._feature_dependency_matrix is None:
            shap_matrix = self.shap_matrix()

            # exclude features with zero Shapley importance
            # noinspection PyUnresolvedReferences
            shap_matrix = shap_matrix.loc[:, (shap_matrix != 0.0).any()]

            self._feature_dependency_matrix = shap_matrix.corr(method="pearson")

        return self._feature_dependency_matrix

    def cluster_dependent_features(self) -> LinkageTree:
        """
        Return the :class:`.LinkageTree` based on the `feature_dependency_matrix`.

        :return: linkage tree for the shap clustering dendrogram
        """
        # convert shap correlations to distances (1 = most distant)
        feature_distance_matrix = 1 - self.feature_dependency_matrix().abs()

        # compress the distance matrix (required by SciPy)
        compressed_distance_vector = squareform(feature_distance_matrix)

        # calculate the linkage matrix
        linkage_matrix = linkage(y=compressed_distance_vector, method="single")

        # feature labels and weights will be used as the leaves of the linkage tree
        feature_importances = self.feature_importances()

        # select only the features that appear in the distance matrix, and in the
        # correct order
        feature_importances = feature_importances.loc[
            feature_importances.index.intersection(feature_distance_matrix.index)
        ]

        # build and return the linkage tree
        return LinkageTree(
            scipy_linkage_matrix=linkage_matrix,
            leaf_labels=feature_importances.index,
            leaf_weights=feature_importances.values,
            max_distance=1.0,
        )


def tree_explainer_factory(estimator: BaseEstimator, data: pd.DataFrame) -> Explainer:
    """
    Return the  explainer :class:`shap.Explainer` used to compute the shap values.

    Try to return :class:`shap.TreeExplainer` if ``self.estimator`` is compatible,
    i.e. is tree-based.
    Otherwise return :class:`shap.KernelExplainer` which is expected to be much slower.

    :param estimator: estimator from which we want to compute shap values
    :param data: data used to compute the shap values
    :return: :class:`shap.TreeExplainer` if the estimator is compatible,
        else :class:`shap.KernelExplainer`."""

    # NOTE:
    # unfortunately, there is no convenient function in shap to determine the best
    # explainer calibration. hence we use this try/except approach.
    # further there is no consistent "ModelPipelineDF type X is unsupported"
    # exception raised,
    # which is why we need to always assume the error resulted from this cause -
    # we should not attempt to filter the exception type or message given that it is
    # currently inconsistent

    try:
        return TreeExplainer(model=estimator)
    except Exception as e:
        log.debug(
            f"failed to instantiate shap.TreeExplainer:{str(e)},"
            "using shap.KernelExplainer as fallback"
        )
        # when using KernelExplainer, shap expects "pipeline" to be a callable that
        # predicts
        # noinspection PyUnresolvedReferences
        return KernelExplainer(model=estimator.predict, data=data)


class RegressorInspector(
    BaseLearnerInspector[_T_RegressorPipelineDF], Generic[_T_RegressorPipelineDF]
):
    """
    Inspect a regression pipeline through its SHAP values.

    :param crossfit: regressor containing the information about the pipeline, \
        the data (a Sample object), the cross-validation and crossfit.
    :param explainer_factory: calibration that returns a shap Explainer
    """

    def __init__(
        self,
        crossfit: RegressorCrossfit[_T_RegressorPipelineDF],
        explainer_factory: Optional[
            Callable[[BaseEstimator, pd.DataFrame], Explainer]
        ] = None,
    ) -> None:
        super().__init__(crossfit=crossfit, explainer_factory=explainer_factory)

    def _shap_matrix_for_split_to_df(
        self,
        raw_shap_values: Union[np.ndarray, List[np.ndarray]],
        split_transformed: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Convert the SHAP matrix for a single split to a data frame.

        :param raw_shap_values: the raw values returned by the SHAP explainer
        :param split_transformed: the transformed data the pipeline was trained on
        :return: SHAP matrix of a single split as data frame
        """

        # In the regression case, only ndarray outputs are expected from SHAP:
        if not isinstance(raw_shap_values, np.ndarray):
            raise ValueError(
                "shap explainer output expected to be an ndarray but was "
                f"{type(raw_shap_values)}"
            )

        return pd.DataFrame(
            data=raw_shap_values,
            index=split_transformed.index,
            columns=split_transformed.columns,
        )


class ClassifierInspector(
    BaseLearnerInspector[_T_ClassifierPipelineDF], Generic[_T_ClassifierPipelineDF]
):
    """
    Inspect a classification pipeline through its SHAP values.

    Currently only binary, single-output classification problems are supported.

    :param crossfit: classifier containing the information about the pipeline, \
        the data (a Sample object), the cross-validation and crossfit.
    :param explainer_factory: calibration that returns a shap Explainer
    """

    def __init__(
        self,
        crossfit: ClassifierCrossfit[_T_ClassifierPipelineDF],
        explainer_factory: Optional[
            Callable[[BaseEstimator, pd.DataFrame], Explainer]
        ] = None,
    ) -> None:
        super().__init__(crossfit=crossfit, explainer_factory=explainer_factory)

    def _shap_matrix_for_split_to_df(
        self,
        raw_shap_values: Union[np.ndarray, List[np.ndarray]],
        split_transformed: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Convert the SHAP matrix for a single split to a data frame.

        :param raw_shap_values: the raw values returned by the SHAP explainer
        :param split_transformed: the transformed data the pipeline was trained on
        :return: SHAP matrix of a single split as data frame
        """

        # todo: adapt this calibration (and override others) to support non-binary
        #   classification

        if isinstance(raw_shap_values, list):
            # the shap explainer returned an array [obs x features] for each of the
            # target-classes

            n_arrays = len(raw_shap_values)

            # we decided to support only binary classification == 2 classes:
            assert n_arrays == 2, (
                "classification pipeline inspection only supports binary classifiers, "
                f"but SHAP analysis returned values for {n_arrays} classes"
            )

            # in the binary classification case, we will proceed with SHAP values
            # for class 0, since values for class 1 will just be the same
            # values times (*-1)  (the opposite probability)

            # to assure the values are returned as expected above,
            # and no information of class 1 is discarded, assert the
            # following:
            assert np.all(
                (raw_shap_values[0]) - (raw_shap_values[1] * -1) < 1e-10
            ), "Expected shap_values(class 0) == shap_values(class 1) * -1"

            # all good: proceed with SHAP values for class 0:
            raw_shap_values = raw_shap_values[0]

        # after the above transformation, `raw_shap_values` should be ndarray:
        if not isinstance(raw_shap_values, np.ndarray):
            raise ValueError(
                f"shap explainer output expected to be an ndarray but was "
                f"{type(raw_shap_values)}"
            )

        return pd.DataFrame(
            data=raw_shap_values,
            index=split_transformed.index,
            columns=split_transformed.columns,
        )
