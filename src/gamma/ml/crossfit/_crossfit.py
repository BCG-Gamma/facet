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
from sklearn.base import BaseEstimator
from sklearn.metrics import check_scoring
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils import check_random_state

from gamma.common.fit import FittableMixin, T_Self
from gamma.common.parallelization import ParallelizableMixin
from gamma.ml import Sample
from gamma.sklearndf import BaseLearnerDF, TransformerDF
from gamma.sklearndf.pipeline import (
    BaseLearnerPipelineDF,
    ClassifierPipelineDF,
    RegressorPipelineDF,
)

log = logging.getLogger(__name__)

__all__ = ["CrossfitScores", "LearnerCrossfit", "Scorer"]

T_LearnerPipelineDF = TypeVar("T_LearnerPipelineDF", bound=BaseLearnerPipelineDF)
T_ClassifierPipelineDF = TypeVar("T_ClassifierPipelineDF", bound=ClassifierPipelineDF)
T_RegressorPipelineDF = TypeVar("T_RegressorPipelineDF", bound=RegressorPipelineDF)

_INDEX_SENTINEL = pd.Index([])

#: a scorer generated by :meth:`sklearn.metrics.make_scorer`
Scorer = Callable[
    [
        # trained learner to use for scoring
        BaseEstimator,
        # test data that will be fed to the learner
        pd.DataFrame,
        # target values for X
        Union[pd.Series, pd.DataFrame],
        # sample weights
        Optional[pd.Series],
    ],
    # result of applying score function to estimator applied to X
    float,
]


class CrossfitScores:
    """"
    Distribution of scores across all cross-validation fits `(crossfits)` of a
    learner pipeline.

    Generated by method :meth:`.LearnerCrossfit.score`.

    Scores for individual fits can be accessed by iteration, or by indexing
    (``[…]`` notation).

    :param scores: list or 1d array of scores for all crossfits of a pipeline
    """

    def __init__(self, scores: Union[Sequence[float], np.ndarray]):
        if isinstance(scores, list):
            scores = np.array(scores)

        if (
            not isinstance(scores, np.ndarray)
            or scores.dtype != float
            or scores.ndim != 1
        ):
            raise TypeError("arg scores must be a list or 1d numpy array of floats")

        self._scores = np.array(scores)

    def __getitem__(self, item: Union[int, slice]) -> Union[float, np.ndarray]:
        return self._scores[item]

    def mean(self) -> float:
        """:return: the mean score"""
        return self._scores.mean()

    def std(self) -> float:
        """:return: the standard deviation of the scores"""
        return self._scores.std()


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
    with optional feature shuffling.

    Feature shuffling can be helpful when fitting models with a data set that contains
    very similar features.
    For such groups of similar features, some learners may pick features based on their
    relative position in the training data table.
    Feature shuffling randomizes the sequence of features for each cross-validation
    training sample, thus ensuring that all similar features have the same chance of
    being used across crossfits.

    Feature shuffling is active by default, so that every model is trained on a random
    permutation of the feature columns to avoid favouring one of several similar
    features based on column sequence.
    """

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
        :param cv: the cross-validator generating the train splits
        :param shuffle_features: if ``True``, shuffle column order of features for \
            every crossfit (default: ``False``)
        :param random_state: optional random seed or random state for shuffling the \
            feature column order
        """
        super().__init__(
            n_jobs=n_jobs,
            shared_memory=shared_memory,
            pre_dispatch=pre_dispatch,
            verbose=verbose,
        )

        self.pipeline: T_LearnerPipelineDF = pipeline.clone()
        self.cv = cv
        self.shuffle_features: bool = (
            False if shuffle_features is None else shuffle_features
        )
        self.random_state = random_state

        self._model_by_split: Optional[List[T_LearnerPipelineDF]] = None
        self._training_sample: Optional[Sample] = None

    __init__.__doc__ += ParallelizableMixin.__init__.__doc__

    def fit(
        self: T_Self,
        sample: Sample,
        sample_weight: Optional[pd.Series] = None,
        **fit_params,
    ) -> T_Self:
        """
        Fit the underlying pipeline to the full sample, and fit clones of the pipeline
        to each of the train splits generated by the cross-validator.

        :param sample: the sample to fit the estimators to
        :param sample_weight: optional weights for all observations in the training \
            sample used to fit this crossfit
        :param fit_params: optional fit parameters, to be passed on to the fit method \
            of the base estimator
        :return: ``self``
        """

        self: LearnerCrossfit  # support type hinting in PyCharm

        self._fit_score(_sample=sample, sample_weight=sample_weight, **fit_params)

        return self

    def score(
        self,
        scoring: Union[str, Callable[[float, float], float], None] = None,
        train_scores: bool = False,
        sample_weight: Optional[pd.Series] = None,
    ) -> CrossfitScores:
        """
        Score all models in this crossfit using the given scoring function.

        The crossfit must already be fitted, see :meth:`.fit`

        :param scoring: scoring to use to score the models (see \
            :meth:`sklearn.metrics.check_scoring` for details)
        :param train_scores: if ``True``, calculate train scores instead of test \
            scores (default: ``False``)
        :param sample_weight: optional weights for all observations in the training \
            sample, to be passed on to the scoring function
        :return: the resulting scores
        """

        return self._fit_score(
            _scoring=scoring, _train_scores=train_scores, sample_weight=sample_weight
        )

    def fit_score(
        self,
        sample: Sample,
        scoring: Union[str, Callable[[float, float], float], None] = None,
        train_scores: bool = False,
        sample_weight: Optional[pd.Series] = None,
        **fit_params,
    ) -> CrossfitScores:
        """
        Fit then score this crossfit.

        See :meth:`.fit` and :meth:`.score` for details.

        :param sample: the sample to which to fit the pipeline underlying this crossfit
        :param fit_params: optional fit parameters, to be passed on to the fit method \
            of the learner
        :param scoring: scoring function to use to score the models \
            (see :meth:`~sklearn.metrics.check_scoring` for details)
        :param train_scores: if ``True``, calculate train scores instead of test \
            scores (default: ``False``)
        :param sample_weight: optional weights for all observations in the sample

        :return: the resulting scores
        """
        return self._fit_score(
            _sample=sample,
            _scoring=scoring,
            _train_scores=train_scores,
            sample_weight=sample_weight,
            **fit_params,
        )

    # noinspection PyPep8Naming
    def _fit_score(
        self,
        _sample: Optional[Sample] = None,
        _scoring: Union[str, Callable[[float, float], float], None] = __NO_SCORING,
        _train_scores: bool = False,
        sample_weight: Optional[pd.Series] = None,
        **fit_params,
    ) -> Optional[CrossfitScores]:

        do_fit = _sample is not None
        do_score = _scoring is not LearnerCrossfit.__NO_SCORING

        assert do_fit or do_score, "at least one of fitting or scoring is enabled"

        pipeline = self.pipeline

        if sample_weight is not None:
            fit_params.update(sample_weight=sample_weight)

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
                    "use a single scorer instead; "
                    f"arg scoring={_scoring} was passed"
                )

            scorer = check_scoring(
                estimator=self.pipeline.final_estimator.root_estimator, scoring=_scoring
            )
        else:
            scorer = None

        def _generate_parameters() -> Iterator[_FitScoreParameters]:
            learner_features = pipeline.features_out
            n_learner_features = len(learner_features)
            test_scores = do_score and not _train_scores
            models = iter(lambda: None, 0) if do_fit else self.models()
            random_state = check_random_state(self.random_state)
            weigh_samples = sample_weight is not None

            for (train, test), model in zip(
                self.cv.split(X=features, y=target), models
            ):
                yield _FitScoreParameters(
                    pipeline=pipeline.clone() if do_fit else model,
                    train_features=(
                        features.iloc[train] if do_fit or _train_scores else None
                    ),
                    train_feature_sequence=(
                        learner_features[random_state.permutation(n_learner_features)]
                        if do_fit and self.shuffle_features
                        else None
                    ),
                    train_target=target.iloc[train] if do_fit else None,
                    train_weight=(
                        sample_weight.iloc[train]
                        if weigh_samples and (do_fit or _train_scores)
                        else None
                    ),
                    scorer=scorer,
                    score_train_split=_train_scores,
                    test_features=features.iloc[test] if test_scores else None,
                    test_target=target.iloc[test] if test_scores else None,
                    test_weight=(
                        sample_weight.iloc[test]
                        if weigh_samples and test_scores
                        else None
                    ),
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

        return CrossfitScores(scores=scores) if do_score else None

    def resize(self: T_Self, n_fits: int) -> T_Self:
        """
        Reduce the size of this crossfit by removing a subset of the fits.
        :param n_fits: the number of fits to keep. Must be lower than the number of \
            fits
        :return:
        """
        self: LearnerCrossfit

        # ensure that arg n_split has a valid value
        if n_fits > self.n_fits:
            raise ValueError(
                f"arg n_fits={n_fits} must not be greater than the number of fits"
                f"in the original crossfit ({self.n_fits} fits)"
            )
        elif n_fits < 1:
            raise ValueError(f"arg n_fits={n_fits} must be a positive integer")

        # copy self and only keep the specified number of fits
        new_crossfit = copy(self)
        new_crossfit._model_by_split = self._model_by_split[:n_fits]
        return new_crossfit

    @property
    def is_fitted(self) -> bool:
        """``True`` if the delegate estimator is fitted, else ``False``"""
        return self._training_sample is not None

    @property
    def n_fits(self) -> int:
        """
        The number of fits in this crossfit.
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

        pipeline: BaseLearnerPipelineDF

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
            preprocessing: TransformerDF = pipeline.preprocessing
            learner: BaseLearnerDF = pipeline.final_estimator

            if parameters.score_train_split:
                features = parameters.train_features
                target = parameters.train_target
            else:
                features = parameters.test_features
                target = parameters.test_target

            if preprocessing:
                features = preprocessing.transform(X=features)

            score = parameters.scorer(
                learner.root_estimator,
                features,
                target,
                fit_params.get("sample_weight", None),
            )

        else:
            score = None

        return pipeline if do_fit else None, score

    def __len__(self) -> int:
        return self.n_fits
