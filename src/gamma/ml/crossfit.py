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
Meta-estimators that fit an estimator multiple times for all splits of a
cross-validator, as the basis for model evaluation and inspection.

:class:`LearnerCrossfit` encapsulates a fully trained pipeline.
It contains a :class:`.ModelPipelineDF` (preprocessing + estimator), a dataset given by a
:class:`yieldengine.Sample` object and a
cross-validation calibration. The pipeline is fitted accordingly.
"""

import logging
from abc import ABC
from typing import *

import pandas as pd
from joblib import delayed, Parallel
from sklearn.model_selection import BaseCrossValidator

from gamma.common import ListLike
from gamma.ml import Sample
from gamma.sklearndf import BaseEstimatorDF, BaseLearnerDF, ClassifierDF, RegressorDF

log = logging.getLogger(__name__)

__all__ = ["BaseCrossfit", "LearnerCrossfit", "RegressorCrossfit", "ClassifierCrossfit"]

_T = TypeVar("_T")
_T_EstimatorDF = TypeVar("_T_EstimatorDF", bound=BaseEstimatorDF)
_T_LearnerDF = TypeVar("_T_LearnerDF", bound=BaseLearnerDF)
_T_ClassifierDF = TypeVar("_T_ClassifierDF", bound=ClassifierDF)
_T_RegressorDF = TypeVar("_T_RegressorDF", bound=RegressorDF)


class BaseCrossfit(ABC, Generic[_T_EstimatorDF]):
    """
    :class:~gamma.sklearn all splits of a given cross-validation
    strategy, based on a pipeline.

    :param base_estimator: predictive pipeline to be fitted
    :param cv: the cross validator generating the train splits
    :param n_jobs: number of jobs to _rank_learners in parallel. Default to ``None`` which is
      interpreted a 1.
    :param shared_memory: if ``True`` use threads in the parallel runs. If `False`
      use multiprocessing
    :param verbose: verbosity level used in the parallel computation
    """

    __slots__ = [
        "base_estimator",
        "cv",
        "n_jobs",
        "shared_memory",
        "verbose",
        "_model_by_split",
    ]

    def __init__(
        self,
        base_estimator: _T_EstimatorDF,
        cv: BaseCrossValidator,
        n_jobs: int = 1,
        shared_memory: bool = True,
        verbose: int = 0,
    ) -> None:
        self.base_estimator = base_estimator
        self.cv = cv
        self.n_jobs = n_jobs
        self.shared_memory = shared_memory
        self.verbose = verbose

        self._model_by_split: Optional[List[_T_EstimatorDF]] = None
        self._training_sample: Optional[Sample] = None

    # noinspection PyPep8Naming
    def fit(self: _T, sample: Sample, **fit_params) -> _T:
        base_estimator = self.base_estimator

        features = sample.features
        target = sample.target

        train_split, test_split = tuple(zip(*self.cv.split(features, target)))

        self._model_by_split: List[_T_EstimatorDF] = self._parallel()(
            delayed(BaseCrossfit._fit_model_for_split)(
                base_estimator.clone(),
                features.iloc[train_indices],
                target.iloc[train_indices],
                **fit_params,
            )
            for train_indices in train_split
        )

        self._training_sample = sample

        return self

    @property
    def is_fitted(self) -> bool:
        """`True` if the delegate estimator is fitted, else `False`"""
        return self._training_sample is not None

    def split(self) -> Generator[Tuple[ListLike[int], ListLike[int]], None, None]:
        self._ensure_fitted()
        return self.cv.split(
            self._training_sample.features, self._training_sample.target
        )

    def get_n_splits(self) -> int:
        """Number of splits used for this crossfit."""
        return self.cv.get_n_splits()

    def models(self) -> Iterator[_T_EstimatorDF]:
        """Iterator of all models fitted on the cross-validation train splits."""
        self._ensure_fitted()
        return iter(self._model_by_split)

    @property
    def training_sample(self) -> Sample:
        """The sample used to train this crossfit."""
        self._ensure_fitted()
        return self._training_sample

    def _ensure_fitted(self) -> None:
        if self._training_sample is None:
            raise RuntimeError(f"{type(self).__name__} expected to be fitted")

    def _parallel(self) -> Parallel:
        return Parallel(
            n_jobs=self.n_jobs,
            require="sharedmem" if self.shared_memory else None,
            verbose=self.verbose,
        )

    # noinspection PyPep8Naming
    @staticmethod
    def _fit_model_for_split(
        estimator: _T_EstimatorDF,
        X: pd.DataFrame,
        y: Union[pd.Series, pd.DataFrame],
        **fit_params,
    ) -> _T_EstimatorDF:
        """
        Fit a pipeline using a sample.

        :param estimator:  the :class:`gamma.ml.ModelPipelineDF` to fit
        :param train_sample: data used to fit the pipeline
        :return: fitted pipeline for the split
        """
        return estimator.fit(X=X, y=y, **fit_params)


class LearnerCrossfit(BaseCrossfit[_T_LearnerDF], Generic[_T_LearnerDF], ABC):
    """
    Generate cross-validated prediction for each observation in a sample, based on
    multiple fits of a learner across a collection of cross-validation splits

    :param base_estimator: predictive pipeline to be fitted
    :param cv: the cross validator generating the train splits
    :param n_jobs: number of jobs to _rank_learners in parallel (default: 1)
    :param shared_memory: if ``True`` use threading in the parallel runs. If `False`, \
      use multiprocessing
    :param verbose: verbosity level used in the parallel computation
    """

    COL_SPLIT_ID = "split_id"
    COL_TARGET = "target"

    def __init__(
        self,
        base_estimator: _T_LearnerDF,
        cv: BaseCrossValidator,
        n_jobs: int = 1,
        shared_memory: bool = True,
        verbose: int = 0,
    ) -> None:
        super().__init__(
            base_estimator=base_estimator,
            cv=cv,
            n_jobs=n_jobs,
            shared_memory=shared_memory,
            verbose=verbose,
        )

    def predictions_oob(
        self, sample: Sample
    ) -> Generator[Union[pd.Series, pd.DataFrame], None, None]:
        """
        Predict all values in the test set.

        The result is a data frame with one row per prediction, indexed by the
        observations in the sample and the split id (index level ``COL_SPLIT_ID``),
        and with columns ``COL_PREDICTION` (the predicted value for the
        given observation and split), and ``COL_TARGET`` (the actual target)

        Note that there can be multiple prediction rows per observation if the test
        splits overlap.

        :return: the data frame with the crossfit per observation and test split
        """

        # todo: move this method to Simulator class -- too specific!

        for split_id, (model, (_, test_indices)) in enumerate(
            zip(self.models(), self.split())
        ):
            test_features = sample.features.iloc[test_indices, :]
            yield model.predict(X=test_features)


class RegressorCrossfit(LearnerCrossfit[_T_RegressorDF], Generic[_T_RegressorDF]):
    pass


class ClassifierCrossfit(LearnerCrossfit[_T_ClassifierDF], Generic[_T_ClassifierDF]):
    __slots__ = ["_probabilities_for_all_samples", "_log_probabilities_for_all_samples"]

    COL_PROBA = "proba_class_0"

    __PROBA = "proba"
    __LOG_PROBA = "log_proba"
    __DECISION_FUNCTION = "decision_function"

    def __init__(
        self,
        base_estimator: _T_ClassifierDF,
        cv: BaseCrossValidator,
        n_jobs: int = 1,
        shared_memory: bool = True,
        verbose: int = 0,
    ):
        super().__init__(
            base_estimator=base_estimator,
            cv=cv,
            n_jobs=n_jobs,
            shared_memory=shared_memory,
            verbose=verbose,
        )

    def probabilities_oob(
        self, sample: Sample
    ) -> Generator[Union[pd.DataFrame, List[pd.DataFrame]], None, None]:
        yield from self._probabilities_oob(
            sample=sample, method=lambda model, x: model.predict_proba(x)
        )

    def log_probabilities_oob(
        self, sample: Sample
    ) -> Generator[Union[pd.DataFrame, List[pd.DataFrame]], None, None]:
        yield from self._probabilities_oob(
            sample=sample, method=lambda model, x: model.predict_log_proba(x)
        )

    def decision_function(
        self, sample: Sample
    ) -> Generator[Union[pd.Series, pd.DataFrame], None, None]:
        yield from self._probabilities_oob(
            sample=sample, method=lambda model, x: model.decision_function(x)
        )

    def _probabilities_oob(
        self,
        sample: Sample,
        method: Callable[
            [_T_ClassifierDF, pd.DataFrame],
            Union[pd.DataFrame, List[pd.DataFrame], pd.Series],
        ],
    ) -> Generator[Union[pd.DataFrame, List[pd.DataFrame], pd.Series], None, None]:
        """
        Predict all values in the test set.

        The result is a data frame with one row per prediction, indexed by the
        observations in the sample and the split id (index level ``COL_SPLIT_ID``),
        and with columns ``COL_PREDICTION` (the predicted value for the
        given observation and split), and ``COL_TARGET`` (the actual target)

        Note that there can be multiple prediction rows per observation if the test
        splits overlap.

        :return: the data frame with the crossfit per observation and test split
        """

        # todo: move this method to Simulator class -- too specific!

        for split_id, (model, (_, test_indices)) in enumerate(
            zip(self.models(), self.split())
        ):
            test_features = sample.features.iloc[test_indices, :]
            yield method(model, test_features)
