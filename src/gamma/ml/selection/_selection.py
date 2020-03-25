"""
Core implementation of :mod:`gamma.ml.selection`
"""

import logging
import re
from abc import ABCMeta
from collections import defaultdict
from itertools import chain
from typing import *

import numpy as np
from numpy.random.mtrand import RandomState
from sklearn.model_selection import BaseCrossValidator, GridSearchCV

from gamma.common.fit import FittableMixin, T_Self
from gamma.common.parallelization import ParallelizableMixin
from gamma.ml import Sample
from gamma.ml.crossfit import LearnerCrossfit
from gamma.sklearndf.pipeline import (
    BaseLearnerPipelineDF,
    ClassifierPipelineDF,
    RegressorPipelineDF,
)

log = logging.getLogger(__name__)

__all__ = [
    "ParameterGrid",
    "Scoring",
    "LearnerEvaluation",
    "BaseLearnerRanker",
    "RegressorRanker",
    "ClassifierRanker",
]

#
# Type variables
#

T_LearnerPipelineDF = TypeVar("T_LearnerPipelineDF", bound=BaseLearnerPipelineDF)
T_RegressorPipelineDF = TypeVar("T_RegressorPipelineDF", bound=RegressorPipelineDF)
T_ClassifierPipelineDF = TypeVar("T_ClassifierPipelineDF", bound=ClassifierPipelineDF)

T_LearnerCrossfit = TypeVar("T_Crossfit", bound=LearnerCrossfit[T_LearnerPipelineDF])

#
# Class definitions
#


class ParameterGrid(Generic[T_LearnerPipelineDF]):
    """
    A grid of hyper-parameters for pipeline tuning.

    :param pipeline: the :class:`ModelPipelineDF` to which the hyper-parameters will \
        be applied
    :param learner_parameters: the hyper-parameter grid in which to search for the \
        optimal parameter values for the pipeline's final estimator
    :param preprocessing_parameters: the hyper-parameter grid in which to search for \
        the optimal parameter values for the pipeline's preprocessing pipeline \
        (optional)
    """

    def __init__(
        self,
        pipeline: T_LearnerPipelineDF,
        learner_parameters: Dict[str, Sequence[Any]],
        preprocessing_parameters: Optional[Dict[str, Sequence[Any]]] = None,
    ) -> None:
        self._pipeline = pipeline
        self._learner_parameters = learner_parameters
        self._preprocessing_parameters = preprocessing_parameters

        def _prefix_parameter_names(
            parameters: Dict[str, Any], prefix: str
        ) -> List[Tuple[str, Any]]:
            return [
                (f"{prefix}__{param}", value) for param, value in parameters.items()
            ]

        grid_parameters: Iterable[Tuple[str, Any]] = _prefix_parameter_names(
            parameters=learner_parameters, prefix=pipeline.final_estimator_name
        )
        if preprocessing_parameters is not None:
            grid_parameters = chain(
                grid_parameters,
                _prefix_parameter_names(
                    parameters=preprocessing_parameters,
                    prefix=pipeline.preprocessing_name,
                ),
            )

        self._grid = dict(grid_parameters)

    @property
    def pipeline(self) -> T_LearnerPipelineDF:
        """
        The :class:`~gamma.ml.EstimatorPipelineDF` for which to optimise the
        parameters.
        """
        return self._pipeline

    @property
    def learner_parameters(self) -> Dict[str, Sequence[Any]]:
        """The parameter grid for the estimator."""
        return self._learner_parameters

    @property
    def preprocessing_parameters(self) -> Optional[Dict[str, Sequence[Any]]]:
        """The parameter grid for the preprocessor."""
        return self._preprocessing_parameters

    @property
    def parameters(self) -> Dict[str, Sequence[Any]]:
        """The parameter grid for the pipeline representing the entire pipeline."""
        return self._grid


class Scoring:
    """"
    Basic statistics on the scoring across all cross validation splits of a pipeline.

    :param split_scores: scores of all cross validation splits for a pipeline
    """

    def __init__(self, split_scores: Iterable[float]):
        self._split_scores = np.array(split_scores)

    def __getitem__(self, item: Union[int, slice]) -> Union[float, np.ndarray]:
        return self._split_scores[item]

    def mean(self) -> float:
        """:return: mean of the split scores"""
        return self._split_scores.mean()

    def std(self) -> float:
        """:return: standard deviation of the split scores"""
        return self._split_scores.std()


class LearnerEvaluation(Generic[T_LearnerPipelineDF]):
    """
    LearnerEvaluation result for a specific parametrisation of a
    learner pipeline, determined by a learner ranker.

    :param pipeline: the unfitted learner pipeline
    :param parameters: the hyper-parameters selected for the learner during grid \
        search, as a mapping of parameter names to parameter values
    :param scoring: maps score names to :class:`~gamma.ml.Scoring` instances
    :param ranking_score: overall score determined by the 's ranking \
        metric, used for ranking all crossfit
    """

    def __init__(
        self,
        pipeline: T_LearnerPipelineDF,
        parameters: Mapping[str, Any],
        scoring: Mapping[str, Scoring],
        ranking_score: float,
    ) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.parameters = parameters
        self.scoring = scoring
        self.ranking_score = ranking_score


class BaseLearnerRanker(
    ParallelizableMixin,
    FittableMixin[Sample],
    Generic[T_LearnerPipelineDF, T_LearnerCrossfit],
    metaclass=ABCMeta,
):
    """
    Rank different parametrisations of one or more learners using cross-validation.

    Given a list of :class:`~gamma.ml.ParameterGrid`, a cross-validation splitter and a
    scoring function, performs a grid search to find the pipeline and
    hyper-parameters with the best score across all cross-validation splits.

    The actual ranking is calculated when invoking the :meth:`.fit` method with a \
    sample.
    """

    __slots__ = [
        "_grids",
        "_sample",
        "_scoring",
        "_cv",
        "_searchers",
        "_pipeline",
        "_ranking_scorer",
        "_ranking_metric",
        "_ranking",
    ]

    _COL_PARAMETERS = "params"

    def __init__(
        self,
        grid: Union[
            ParameterGrid[T_LearnerPipelineDF],
            Iterable[ParameterGrid[T_LearnerPipelineDF]],
        ],
        cv: Optional[BaseCrossValidator],
        scoring: Union[
            str,
            Callable[[float, float], float],
            List[str],
            Tuple[str],
            Dict[str, Callable[[float, float], float]],
            None,
        ] = None,
        ranking_scorer: Callable[[float, float], float] = None,
        ranking_metric: str = "test_score",
        shuffle_features: Optional[bool] = None,
        random_state: Union[int, RandomState, None] = None,
        n_jobs: Optional[int] = None,
        shared_memory: bool = False,
        pre_dispatch: str = "2*n_jobs",
        verbose: int = 0,
    ) -> None:
        """
        :param grid: :class:`~gamma.ml.ParameterGrid` to be ranked \
            (either a single grid, or an iterable of multiple grids)
        :param cv: a cross validator (e.g., \
            :class:`~gamma.ml.validation.BootstrapCV`)
        :param scoring: a scorer to use when doing CV within GridSearch, defaults to \
            `None`
        :param ranking_scorer: scoring function used for ranking across crossfit, \
            taking mean and standard deviation of the ranking scores_for_split and \
            returning the overall ranking score \
            (default: :meth:`.default_ranking_scorer`)
        :param ranking_metric: the scoring to be used for pipeline ranking, \
            given as a name to be used to look up the right Scoring object in the \
            LearnerEvaluation.scoring dictionary (default: 'test_score').
        :param shuffle_features: if `True`, shuffle column order of features for every \
            crossfit (default: `False`)
        :param random_state: optional random seed or random state for shuffling the \
            feature column order
        :param n_jobs: number of jobs to use in parallel; \
            if `None`, use joblib default (default: `None`).
        :param shared_memory: if `True` use threads in the parallel runs. If `False` \
            use multiprocessing (default: `False`).
        :param pre_dispatch: number of batches to pre-dispatch; \
            if `None`, use joblib default (default: `None`).
        :param verbose: verbosity level used in the parallel computation; \
            if `None`, use joblib default (default: `None`).
        """
        super().__init__(
            n_jobs=n_jobs,
            shared_memory=shared_memory,
            pre_dispatch=pre_dispatch,
            verbose=verbose,
        )
        self._grids = list(grid) if isinstance(grid, Iterable) else [grid]
        self._cv = cv
        self._scoring = scoring
        self._ranking_scorer = (
            BaseLearnerRanker.default_ranking_scorer
            if ranking_scorer is None
            else ranking_scorer
        )
        self._ranking_metric = ranking_metric
        self._shuffle_features = shuffle_features
        self._random_state = random_state

        # initialise state
        self._sample: Optional[Sample] = None
        self._fit_params: Optional[Dict[str, Any]] = None
        self._ranking: Optional[List[LearnerEvaluation]] = None

    @staticmethod
    def default_ranking_scorer(scoring: Scoring) -> float:
        """
        The default function to determine the pipeline's rank: ``mean - 2 * std``.

        Its output is used to rank different parametrizations of one or more learners.

        :param scoring: the :class:`Scoring` with validation scores for a given split
        :return: score to be used for pipeline ranking
        """
        return scoring.mean() - 2 * scoring.std()

    def fit(self: T_Self, sample: Sample, **fit_params) -> T_Self:
        """
        Rank the candidate learners and their hyper-parameter combinations using the
        given sample.

        :param sample: sample with which to fit the candidate learners from the grid(s)
        :param fit_params: any fit parameters to pass on to the learner's fit method
        """
        self: BaseLearnerRanker  # support type hinting in PyCharm
        self._rank_learners(sample=sample, **fit_params)
        return self

    @property
    def is_fitted(self) -> bool:
        """`True` if this ranker is fitted, `False` otherwise."""
        return self._sample is not None

    def ranking(self) -> List[LearnerEvaluation[T_LearnerPipelineDF]]:
        """
        :return a ranking of all learners that were evaluated based on the parameter
        grids passed to this ranker, in descending order of the ranking score.
        """
        self._ensure_fitted()
        return self._ranking.copy()

    @property
    def best_model(self) -> T_LearnerPipelineDF:
        """
        The pipeline which obtained the best ranking score, fitted on the entire sample
        """
        return self._best_pipeline().fit(X=self._sample.features, y=self._sample.target)

    @property
    def best_model_crossfit(self,) -> T_LearnerCrossfit:
        """
        The crossfit for the best model, fitted with the same sample and fit
        parameters used to fit this ranker.
        """

        return LearnerCrossfit(
            pipeline=self._best_pipeline(),
            cv=self._cv,
            shuffle_features=self._shuffle_features,
            random_state=self._random_state,
            n_jobs=self.n_jobs,
            shared_memory=self.shared_memory,
            pre_dispatch=self.pre_dispatch,
            verbose=self.verbose,
        ).fit(sample=self._sample, **self._fit_params)

    def summary_report(self, max_learners: Optional[int] = None) -> str:
        """
        Return a human-readable report of learner validation results, sorted by
        ranking score in descending order.

        :param max_learners: maximum number of learners to include in the report \
            (optional)

        :return: a summary string of the pipeline ranking
        """

        self._ensure_fitted()

        def _model_name(evaluation: LearnerEvaluation) -> str:
            return type(evaluation.pipeline.final_estimator).__name__

        def _parameters(params: Mapping[str, Iterable[Any]]) -> str:
            return ",".join(
                [
                    f"{param_name}={param_value}"
                    for param_name, param_value in params.items()
                ]
            )

        def _score_summary(scoring_dict: Mapping[str, Scoring]) -> str:
            return ", ".join(
                [
                    f"{score}_mean={scoring.mean():9.3g}, "
                    f"{score}_std={scoring.std():9.3g}, "
                    for score, scoring in sorted(
                        scoring_dict.items(), key=lambda pair: pair[0]
                    )
                ]
            )

        ranking = self._ranking[:max_learners] if max_learners else self._ranking

        name_width = max([len(_model_name(ranked_model)) for ranked_model in ranking])

        return "\n".join(
            [
                f"Rank {rank + 1:2d}: "
                f"{_model_name(evaluation):>{name_width}s}, "
                f"Score={evaluation.ranking_score:9.3g}, "
                f"{_score_summary(evaluation.scoring)}, "
                f"Parameters={{{_parameters(evaluation.parameters)}}}"
                "\n"
                for rank, evaluation in enumerate(ranking)
            ]
        )

    def _best_pipeline(self) -> T_LearnerPipelineDF:
        # return the unfitted model with the best parametrisation
        self._ensure_fitted()
        return self._ranking[0].pipeline

    def _rank_learners(self, sample: Sample, **fit_params) -> None:

        if len(fit_params) > 0:
            log.warning(
                "Ignoring arg fit_params: current ranker implementation uses "
                "GridSearchCV which does not support fit_params"
            )

        ranking_scorer = self._ranking_scorer

        # construct searchers
        searchers: List[Tuple[GridSearchCV, ParameterGrid]] = [
            (
                GridSearchCV(
                    estimator=grid.pipeline,
                    param_grid=grid.parameters,
                    scoring=self._scoring,
                    n_jobs=self.n_jobs,
                    iid=False,
                    refit=False,
                    cv=self._cv,
                    verbose=self.verbose,
                    pre_dispatch=self.pre_dispatch,
                    return_train_score=False,
                ),
                grid,
            )
            for grid in self._grids
        ]

        for searcher, _ in searchers:
            searcher.fit(X=sample.features, y=sample.target)

        #
        # consolidate results of all searchers into "results"
        #

        def _scoring(
            cv_results: Mapping[str, Sequence[float]]
        ) -> List[Dict[str, Scoring]]:
            """
            Convert ``cv_results_`` into a mapping with :class:`Scoring` values.

            Helper function;  for each pipeline in the grid returns a tuple of test
            scores_for_split across all splits.
            The length of the tuple is equal to the number of splits that were tested
            The test scores_for_split are sorted in the order the splits were tested.

            :param cv_results: a :attr:`sklearn.GridSearchCV.cv_results_` attribute
            :return: a list of test scores per scored pipeline; each list entry maps \
                score types (as str) to a :class:`Scoring` of scores per split. The \
                i-th element of this list is typically of the form \
                ``{'train_score': model_scoring1, 'test_score': model_scoring2,...}``
            """

            # the splits are stored in the cv_results using keys 'split0...'
            # through 'split<nn>...'
            # match these dictionary keys in cv_results; ignore all other keys
            matches_for_split_x_metric: List[Tuple[str, Match]] = [
                (key, re.fullmatch(r"split(\d+)_((train|test)_[a-zA-Z0-9]+)", key))
                for key in cv_results.keys()
            ]

            # extract the integer indices from the matched results keys
            # create tuples (metric, split_index, scores_per_model_for_split),
            # e.g., ('test_r2', 0, [0.34, 0.23, ...])
            metric_x_split_index_x_scores_per_model: List[
                Tuple[str, int, Sequence[float]]
            ] = sorted(
                (
                    (match.group(2), int(match.group(1)), cv_results[key])
                    for key, match in matches_for_split_x_metric
                    if match is not None
                ),
                key=lambda x: x[1],  # sort by split_id so we can later collect scores
                # in the correct sequence
            )

            # Group results per pipeline, result is a list where each item contains the
            # scoring for one pipeline. Each scoring is a dictionary, mapping each
            # metric to a list of scores for the different splits.
            n_models = len(cv_results[BaseLearnerRanker._COL_PARAMETERS])

            scores_per_model_per_metric_per_split: List[Dict[str, List[float]]] = [
                defaultdict(list) for _ in range(n_models)
            ]

            for (
                metric,
                split_ix,
                split_score_per_model,
            ) in metric_x_split_index_x_scores_per_model:
                for model_ix, split_score in enumerate(split_score_per_model):
                    scores_per_model_per_metric_per_split[model_ix][metric].append(
                        split_score
                    )
            # Now in general, the i-th element of scores_per_model_per_metric_per_split
            # is a dict
            # {'train_score': [a_0,...,a_(n-1)], 'test_score': [b_0,..,b_(n-1)]} where
            # a_j (resp. b_j) is the train (resp. test) score for pipeline i in split j

            return [
                {
                    metric: Scoring(split_scores=scores_per_split)
                    for metric, scores_per_split in scores_per_metric_per_split.items()
                }
                for scores_per_metric_per_split in scores_per_model_per_metric_per_split
            ]

        ranking_metric = self._ranking_metric
        ranking = [
            LearnerEvaluation(
                pipeline=grid.pipeline.clone().set_params(**params),
                parameters=params,
                scoring=scoring,
                # compute the final score using the function defined above:
                ranking_score=ranking_scorer(scoring[ranking_metric]),
            )
            for searcher, grid in searchers
            # we read and iterate over these 3 attributes from cv_results_:
            for params, scoring in zip(
                searcher.cv_results_[BaseLearnerRanker._COL_PARAMETERS],
                _scoring(searcher.cv_results_),
            )
        ]

        ranking.sort(key=lambda validation: validation.ranking_score, reverse=True)

        self._sample = sample
        self._fit_params = fit_params
        self._ranking = ranking


class RegressorRanker(
    BaseLearnerRanker[T_RegressorPipelineDF, LearnerCrossfit[T_RegressorPipelineDF]],
    Generic[T_RegressorPipelineDF],
):
    """[inheriting doc string of base class]"""

    __doc__ = cast(str, BaseLearnerRanker.__doc__).replace("learner", "regressor")


class ClassifierRanker(
    BaseLearnerRanker[T_ClassifierPipelineDF, LearnerCrossfit[T_ClassifierPipelineDF]],
    Generic[T_ClassifierPipelineDF],
):
    """[inheriting doc string of base class]"""

    __doc__ = cast(str, BaseLearnerRanker.__doc__).replace("learner", "classifier")
