"""
Core implementation of :mod:`gamma.ml.crossfit`
"""
import logging
from abc import ABCMeta
from copy import copy
from typing import *

import numpy as np
import pandas as pd
from numpy.random.mtrand import RandomState
from sklearn.metrics import check_scoring
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils import check_random_state

from gamma.common.fit import FittableMixin, T_Self
from gamma.common.parallelization import ParallelizableMixin
from gamma.ml import Sample
from gamma.sklearndf import BaseLearnerDF
from gamma.sklearndf.pipeline import (
    BaseLearnerPipelineDF,
    ClassifierPipelineDF,
    RegressorPipelineDF,
)

log = logging.getLogger(__name__)

__all__ = ["LearnerCrossfit", "Scorer", "Scoring"]

T_LearnerPipelineDF = TypeVar("T_LearnerPipelineDF", bound=BaseLearnerPipelineDF)
T_ClassifierPipelineDF = TypeVar("T_ClassifierPipelineDF", bound=ClassifierPipelineDF)
T_RegressorPipelineDF = TypeVar("T_RegressorPipelineDF", bound=RegressorPipelineDF)

_INDEX_SENTINEL = pd.Index([])


Scorer = Callable[
    [
        # trained learner to use for scoring
        BaseLearnerDF,
        # test data that will be fed to the learner
        pd.DataFrame,
        # target values for X
        Union[pd.Series, pd.DataFrame],
        # sample weights
        pd.Series,
    ],
    # result of applying score function to estimator applied to X
    float,
]
#: a scorer generated by :meth:`sklearn.metrics.make_scorer`


class Scoring:
    """"
    Basic statistics on the scoring across all cross validation splits of a pipeline.

    :param split_scores: scores of all cross validation splits for a pipeline
    """

    def __init__(self, split_scores: Sequence[float]):
        self._split_scores = np.array(split_scores)
        assert self._split_scores.dtype == float

    def __getitem__(self, item: Union[int, slice]) -> Union[float, np.ndarray]:
        return self._split_scores[item]

    def mean(self) -> float:
        """:return: mean of the split scores"""
        return self._split_scores.mean()

    def std(self) -> float:
        """:return: standard deviation of the split scores"""
        return self._split_scores.std()


class _FitScoreParameters(NamedTuple):
    pipeline: T_LearnerPipelineDF

    # fit parameters
    train_features: Optional[pd.DataFrame]
    train_feature_sequence: Optional[pd.Index]
    train_target: Union[pd.Series, pd.DataFrame, None]
    train_weight: Optional[pd.Series]

    # score parameters
    scorer: Optional[Scorer]
    score_train_split: bool
    test_features: Optional[pd.DataFrame]
    test_target: Union[pd.Series, pd.DataFrame, None]
    test_weight: Optional[pd.Series]


class LearnerCrossfit(
    FittableMixin[Sample],
    ParallelizableMixin,
    Generic[T_LearnerPipelineDF],
    metaclass=ABCMeta,
):
    """
    Fits a learner pipeline to all train splits of a given cross-validation strategy,
    and with optional feature shuffling.

    Feature shuffling is active by default, so that every model is trained on a random
    permutation of the feature columns to avoid favouring one of several similar
    features based on column sequence.
    """

    __slots__ = [
        "pipeline",
        "cv",
        "n_jobs",
        "shared_memory",
        "verbose",
        "_model_by_split",
    ]

    __NO_SCORING = "<no scoring>"

    def __init__(
        self,
        pipeline: T_LearnerPipelineDF,
        cv: BaseCrossValidator,
        *,
        shuffle_features: Optional[bool] = None,
        random_state: Union[int, RandomState, None] = None,
        n_jobs: Optional[int] = None,
        shared_memory: Optional[bool] = None,
        pre_dispatch: Optional[Union[str, int]] = None,
        verbose: Optional[int] = None,
    ) -> None:
        """
        :param pipeline: learner pipeline to be fitted
        :param cv: the cross validator generating the train splits
        :param shuffle_features: if `True`, shuffle column order of features for every \
            crossfit (default: `False`)
        :param random_state: optional random seed or random state for shuffling the \
            feature column order
        """
        super().__init__(
            n_jobs=n_jobs,
            shared_memory=shared_memory,
            pre_dispatch=pre_dispatch,
            verbose=verbose,
        )
        self.pipeline = pipeline.clone()  #: the learner pipeline being trained
        self.cv = cv  #: the cross validator
        self.shuffle_features: bool = (
            False if shuffle_features is None else shuffle_features
        )
        self.random_state = random_state

        self._model_by_split: Optional[List[T_LearnerPipelineDF]] = None
        self._training_sample: Optional[Sample] = None

    __init__.__doc__ += ParallelizableMixin.__init__.__doc__

    def fit(self: T_Self, sample: Sample, **fit_params) -> T_Self:
        """
        Fit the base estimator to the full sample, and fit a clone of the base
        estimator to each of the train splits generated by the cross-validator
        :param sample: the sample to fit the estimators to
        :param fit_params: optional fit parameters, to be passed on to the fit method \
            of the base estimator
        :return: `self`
        """

        self: LearnerCrossfit  # support type hinting in PyCharm

        self._fit_score(_sample=sample, **fit_params)

        return self

    def score(
        self,
        scoring: Union[str, Callable[[float, float], float], None] = None,
        train_scores: bool = False,
        sample_weight: Optional[pd.Series] = None,
    ) -> Scoring:
        """
        Score all models in this crossfit using the given scoring function

        :param scoring: scoring to use to score the models (see \
            :meth:`~sklearn.metrics.scorer.check_scoring` for details)
        :param train_scores: if `True`, calculate train scores instead of test scores \
            (default: `False`)
        :param sample_weight: optional weights for all observations in the training \
            sample used to fit this crossfit
        :return: the resulting scoring
        """

        return self._fit_score(
            _scoring=scoring, _train_scores=train_scores, _sample_weight=sample_weight
        )

    def fit_score(
        self,
        sample: Sample,
        scoring: Union[str, Callable[[float, float], float], None] = None,
        train_scores: bool = False,
        sample_weight: Optional[pd.Series] = None,
        **fit_params,
    ) -> Scoring:
        """
        Fit and score the base estimator.

        First, fit the base estimator to the full sample, and fit a clone of the base
        estimator to each of the train splits generated by the cross-validator.

        Then, score all models in this crossfit using the given scoring function.

        :param sample: the sample to fit the estimators to
        :param fit_params: optional fit parameters, to be passed on to the fit method \
            of the base estimator
        :param scoring: scoring to use to score the models (see \
            :meth:`~sklearn.metrics.scorer.check_scoring` for details)
        :param train_scores: if `True`, calculate train scores instead of test scores \
            (default: `False`)
        :param sample_weight: optional weights for all observations in the sample

        :return: the resulting scoring
        """
        return self._fit_score(
            _sample=sample,
            _scoring=scoring,
            _train_scores=train_scores,
            _sample_weight=sample_weight,
            **fit_params,
        )

    # noinspection PyPep8Naming
    def _fit_score(
        self,
        _sample: Optional[Sample] = None,
        _scoring: Union[str, Callable[[float, float], float], None] = None,
        _train_scores: bool = False,
        _sample_weight: Optional[pd.Series] = None,
        **fit_params,
    ) -> Optional[Scoring]:

        do_fit = _sample is not None
        do_score = _scoring is not LearnerCrossfit.__NO_SCORING

        pipeline = self.pipeline

        if not do_fit:
            _sample = self.training_sample

        features = _sample.features
        target = _sample.target

        if do_fit:
            pipeline.fit(X=features, y=target, **fit_params)

        # prepare scoring

        scorer: Optional[Scorer]

        if do_score:
            if not isinstance(_scoring, str) and isinstance(_scoring, Container):
                raise ValueError(
                    "Multi-metric scoring is not supported, "
                    "use a single scorer instead. "
                    f"Arg scoring={_scoring} was passed."
                )

            scorer = check_scoring(
                estimator=self.pipeline.final_estimator, scoring=_scoring
            )
        else:
            scorer = None

        def _generate_parameters() -> Iterator[_FitScoreParameters]:
            learner_features = pipeline.features_out
            n_learner_features = len(learner_features)
            test_scores = do_score and not _train_scores
            models = iter(lambda: None, 0) if do_fit else self.models()
            random_state = check_random_state(self.random_state)
            weigh_samples = _sample_weight is not None

            for (train, test), model in zip(
                self.cv.split(X=features, y=target), models
            ):
                yield _FitScoreParameters(
                    pipeline=pipeline.clone() if do_fit else model,
                    train_features=features.iloc[train]
                    if do_fit or _train_scores
                    else None,
                    train_feature_sequence=learner_features[
                        random_state.permutation(n_learner_features)
                    ]
                    if do_fit and self.shuffle_features
                    else None,
                    train_target=target.iloc[train] if do_fit else None,
                    train_weight=_sample_weight.iloc[train]
                    if weigh_samples and (do_fit or _train_scores)
                    else None,
                    scorer=scorer,
                    score_train_split=_train_scores,
                    test_features=features.iloc[test] if test_scores else None,
                    test_target=target.iloc[test] if test_scores else None,
                    test_weight=_sample_weight.iloc[test]
                    if weigh_samples and test_scores
                    else None,
                )

        with self._parallel() as parallel:
            model_and_score_by_split: List[
                Tuple[T_LearnerPipelineDF, Optional[float]]
            ] = parallel(
                self._delayed(LearnerCrossfit._fit_and_score_model_for_split)(
                    parameters, **fit_params
                )
                for parameters in _generate_parameters()
            )

        model_by_split, scores = (
            list(items) for items in zip(*model_and_score_by_split)
        )

        if do_fit:
            self._model_by_split = model_by_split
            self._training_sample = _sample

        return Scoring(split_scores=scores) if do_score else None

    def resize(self: T_Self, n_splits: int) -> T_Self:
        """
        Reduce the size of this crossfit by removing a subset of the fits.
        :param n_splits: the number of fits to keep. Must be lower than the number of
            fits
        :return:
        """
        self: LearnerCrossfit

        # ensure that arg n_split has a valid value
        if n_splits > self.get_n_splits():
            raise ValueError(
                f"arg n_splits={n_splits} must not be greater than the number of splits"
                f"in the original crossfit ({self.get_n_splits()} splits)"
            )
        elif n_splits < 1:
            raise ValueError(f"arg n_splits={n_splits} must be a positive integer")

        # copy self and only keep the specified number of fits
        new_crossfit = copy(self)
        new_crossfit._model_by_split = self._model_by_split[:n_splits]
        return new_crossfit

    @property
    def is_fitted(self) -> bool:
        """`True` if the delegate estimator is fitted, else `False`"""
        return self._training_sample is not None

    def get_n_splits(self) -> int:
        """
        Number of splits used for this crossfit.
        """
        self._ensure_fitted()
        return len(self._model_by_split)

    def splits(self) -> Iterator[Tuple[Sequence[int], Sequence[int]]]:
        """
        :return: an iterator of all train/test splits used by this crossfit
        """
        self._ensure_fitted()

        # ensure we do not return more splits than we have fitted models
        # this is relevant if this is a resized learner crossfit
        return (
            s
            for s, _ in zip(
                self.cv.split(
                    X=self._training_sample.features, y=self._training_sample.target
                ),
                self._model_by_split,
            )
        )

    def models(self) -> Iterator[T_LearnerPipelineDF]:
        """Iterator of all models fitted on the cross-validation train splits."""
        self._ensure_fitted()
        return iter(self._model_by_split)

    @property
    def training_sample(self) -> Sample:
        """The sample used to train this crossfit."""
        self._ensure_fitted()
        return self._training_sample

    # noinspection PyPep8Naming
    @staticmethod
    def _fit_and_score_model_for_split(
        parameters: _FitScoreParameters, **fit_params
    ) -> Tuple[Optional[T_LearnerPipelineDF], Optional[float]]:
        do_fit = parameters.train_target is not None
        do_score = parameters.scorer is not None

        if do_fit:
            pipeline = parameters.pipeline.fit(
                X=parameters.train_features,
                y=parameters.train_target,
                feature_sequence=parameters.train_feature_sequence,
                **fit_params,
            )
        else:
            pipeline = parameters.pipeline

        score: Optional[float]
        if do_score:
            score = parameters.scorer(
                pipeline,
                parameters.train_features
                if parameters.score_train_split
                else parameters.test_features,
                parameters.train_target
                if parameters.score_train_split
                else parameters.test_target,
            )
        else:
            score = None

        return pipeline if do_fit else None, score
