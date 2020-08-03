"""
Core implementation of :mod:`gamma.ml.selection`
"""

import logging
import math
import operator
from functools import reduce
from itertools import chain
from typing import *

from numpy.random.mtrand import RandomState
from sklearn.model_selection import BaseCrossValidator

from gamma.common import to_tuple
from gamma.common.fit import FittableMixin, T_Self
from gamma.common.parallelization import ParallelizableMixin
from gamma.ml import Sample
from gamma.ml.crossfit import CrossfitScores, LearnerCrossfit
from gamma.sklearndf.pipeline import (
    BaseLearnerPipelineDF,
    ClassifierPipelineDF,
    RegressorPipelineDF,
)

log = logging.getLogger(__name__)

__all__ = ["ParameterGrid", "LearnerEvaluation", "LearnerRanker"]

#
# Type variables
#

T_LearnerPipelineDF = TypeVar("T_LearnerPipelineDF", bound=BaseLearnerPipelineDF)
T_RegressorPipelineDF = TypeVar("T_RegressorPipelineDF", bound=RegressorPipelineDF)
T_ClassifierPipelineDF = TypeVar("T_ClassifierPipelineDF", bound=ClassifierPipelineDF)

#
# Class definitions
#


class ParameterGrid(Sequence[Dict[str, Any]], Generic[T_LearnerPipelineDF]):
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
        learner_parameters: Dict[str, Sequence],
        preprocessing_parameters: Optional[Dict[str, Sequence]] = None,
    ) -> None:
        self._pipeline = pipeline
        self._learner_parameters = learner_parameters
        self._preprocessing_parameters = preprocessing_parameters

        def _prefix_parameter_names(
            parameters: Dict[str, Sequence], prefix: str
        ) -> Iterable[Tuple[str, Any]]:
            return (
                (f"{prefix}__{param}", values) for param, values in parameters.items()
            )

        grid_parameters: Iterable[Tuple[str, Sequence]] = _prefix_parameter_names(
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

        self._grid_parameters: List[Tuple[str, Sequence]] = list(grid_parameters)
        self._grid_dict: Dict[str, Sequence] = dict(self._grid_parameters)

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
        return self._grid_dict

    def __iter__(self) -> Iterable[Dict[str, Any]]:
        grid = self._grid_parameters
        params: List[Tuple[str, Any]] = [("", None) for _ in grid]

        def _iter_parameter(param_index: int):
            if param_index < 0:
                yield dict(params)
            else:
                name, values = grid[param_index]
                for value in values:
                    params[param_index] = (name, value)
                    yield from _iter_parameter(param_index=param_index - 1)

        yield from _iter_parameter(len(grid) - 1)

    def __getitem__(
        self, pos: Union[int, slice]
    ) -> Union[Dict[str, Sequence], Sequence[Dict[str, Sequence]]]:

        _len = len(self)

        def _get(i: int) -> Dict[str, Sequence]:
            assert i >= 0

            parameters = self._grid_parameters
            result: Dict[str, Sequence] = {}

            for name, values in parameters:
                n_values = len(values)
                result[name] = values[i % n_values]
                i //= n_values

            assert i == 0

            return result

        def _clip(i: int, i_max: int) -> int:
            if i < 0:
                return max(_len + i, 0)
            else:
                return min(i, i_max)

        if isinstance(pos, slice):
            print(pos)
            return [
                _get(i)
                for i in range(
                    _clip(pos.start or 0, _len - 1),
                    _clip(pos.stop or _len, _len),
                    pos.step or 1,
                )
            ]
        else:
            if pos < -_len or pos >= _len:
                raise ValueError(f"index out of bounds: {pos}")
            return _get(_len + pos if pos < 0 else pos)

    def __len__(self) -> int:
        return reduce(
            operator.mul,
            (
                len(values_for_parameter)
                for values_for_parameter in self._grid_dict.values()
            ),
        )


class LearnerEvaluation(Generic[T_LearnerPipelineDF]):
    """
    LearnerEvaluation result for a specific parametrisation of a
    learner pipeline, determined by a learner ranker.

    :param pipeline: the unfitted learner pipeline
    :param parameters: the hyper-parameters selected for the learner during grid \
        search, as a mapping of parameter names to parameter values
    :param scoring: maps score names to :class:`~gamma.ml.CrossfitScores` instances
    :param ranking_score: overall score determined by the ranking \
        metric, used for ranking all crossfits
    """

    def __init__(
        self,
        pipeline: T_LearnerPipelineDF,
        parameters: Mapping[str, Any],
        scoring: Mapping[str, CrossfitScores],
        ranking_score: float,
    ) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.parameters = parameters
        self.scoring = scoring
        self.ranking_score = ranking_score


class LearnerRanker(
    ParallelizableMixin, FittableMixin[Sample], Generic[T_LearnerPipelineDF]
):
    """
    Rank different parametrizations of one or more learners using cross-validation.

    Given a list of :class:`.ParameterGrid`, a cross-validation splitter and a
    scoring function, performs a grid search to find the pipeline and
    hyper-parameters with the best score across all cross-validation splits.

    The actual ranking is calculated when invoking the :meth:`.fit` method with a \
    sample.
    """

    TEST_SCORE_NAME = "test_score"

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
        shared_memory: Optional[bool] = None,
        pre_dispatch: Optional[Union[str, int]] = None,
        verbose: Optional[int] = None,
    ) -> None:
        """
        :param grid: :class:`~gamma.ml.ParameterGrid` to be ranked \
            (either a single grid, or an iterable of multiple grids)
        :param cv: a cross validator (e.g., \
            :class:`~gamma.ml.validation.BootstrapCV`)
        :param scoring: a scorer to use when doing CV within GridSearch, defaults to \
            ``None``
        :param ranking_scorer: scoring function used for ranking across crossfit, \
            taking mean and standard deviation of the ranking scores_for_split and \
            returning the overall ranking score \
            (default: :meth:`.default_ranking_scorer`)
        :param ranking_metric: the scoring to be used for pipeline ranking, \
            given as a name to be used to look up the right CrossfitScores object in the \
            LearnerEvaluation.scoring dictionary (default: 'test_score').
        :param shuffle_features: if ``True``, shuffle column order of features for every \
            crossfit (default: ``False``)
        :param random_state: optional random seed or random state for shuffling the \
            feature column order
        """
        super().__init__(
            n_jobs=n_jobs,
            shared_memory=shared_memory,
            pre_dispatch=pre_dispatch,
            verbose=verbose,
        )

        self._grids: Tuple[ParameterGrid, ...] = to_tuple(
            grid, element_type=ParameterGrid, arg_name="grid"
        )
        self._cv = cv
        self._scoring = scoring
        self._ranking_scorer = (
            LearnerRanker.default_ranking_scorer
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

    # add parameter documentation of ParallelizableMixin
    __init__.__doc__ += ParallelizableMixin.__init__.__doc__

    @staticmethod
    def default_ranking_scorer(scores: CrossfitScores) -> float:
        """
        The default function to determine the pipeline's rank: ``mean - 2 * std``.

        Its output is used to rank different parametrizations of one or more learners.

        :param scores: the validation scores for all crossfits
        :return: scalar score to be used for ranking the pipeline
        """
        return scores.mean() - 2 * scores.std()

    def fit(self: T_Self, sample: Sample, **fit_params) -> T_Self:
        """
        Rank the candidate learners and their hyper-parameter combinations using the
        given sample.

        :param sample: sample with which to fit the candidate learners from the grid(s)
        :param fit_params: any fit parameters to pass on to the learner's fit method
        """
        self: LearnerRanker[T_LearnerPipelineDF]  # support type hinting in PyCharm

        ranking: List[LearnerEvaluation[T_LearnerPipelineDF]] = self._rank_learners(
            sample=sample, **fit_params
        )
        ranking.sort(key=lambda le: le.ranking_score, reverse=True)

        self._sample = sample
        self._fit_params = fit_params
        self._ranking = ranking

        return self

    @property
    def is_fitted(self) -> bool:
        """``True`` if this ranker is fitted, ``False`` otherwise."""
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
        self._ensure_fitted()
        return self._best_pipeline().fit(X=self._sample.features, y=self._sample.target)

    @property
    def best_model_crossfit(self) -> LearnerCrossfit[T_LearnerPipelineDF]:
        """
        The crossfit for the best model, fitted with the same sample and fit
        parameters used to fit this ranker.
        """
        self._ensure_fitted()
        return self._best_crossfit

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

        def _score_summary(scoring_dict: Mapping[str, CrossfitScores]) -> str:
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

    def _rank_learners(
        self, sample: Sample, **fit_params
    ) -> List[LearnerEvaluation[T_LearnerPipelineDF]]:
        ranking_scorer = self._ranking_scorer

        ranking_metric = self._ranking_metric
        if ranking_metric is not None and ranking_metric != self.TEST_SCORE_NAME:
            raise ValueError(
                f"unsupported ranking metric {ranking_metric}. Use None instead."
            )

        configurations = (
            (grid.pipeline.clone().set_params(**parameters), parameters)
            for grid in self._grids
            for parameters in grid
        )

        ranking: List[LearnerEvaluation[T_LearnerPipelineDF]] = []
        best_score: float = -math.inf
        best_crossfit: Optional[LearnerCrossfit[T_LearnerPipelineDF]] = None

        for pipeline, parameters in configurations:
            crossfit = LearnerCrossfit(
                pipeline=pipeline,
                cv=self._cv,
                shuffle_features=self._shuffle_features,
                random_state=self._random_state,
                n_jobs=self.n_jobs,
                shared_memory=self.shared_memory,
                pre_dispatch=self.pre_dispatch,
                verbose=self.verbose,
            )

            pipeline_scoring: CrossfitScores = crossfit.fit_score(
                sample=sample, scoring=self._scoring, **fit_params
            )

            ranking_score = ranking_scorer(pipeline_scoring)

            ranking.append(
                LearnerEvaluation(
                    pipeline=pipeline,
                    parameters=parameters,
                    scoring={self.TEST_SCORE_NAME: pipeline_scoring},
                    ranking_score=ranking_score,
                )
            )

            if ranking_score > best_score:
                best_score = ranking_score
                best_crossfit = crossfit

        self._best_crossfit = best_crossfit
        return ranking
