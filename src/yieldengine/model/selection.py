from typing import *

from sklearn.model_selection import BaseCrossValidator, GridSearchCV
from sklearn.model_selection._split import BaseShuffleSplit

from yieldengine import Sample
from yieldengine.pipeline import ModelPipeline


class Model(NamedTuple):
    pipeline: ModelPipeline
    parameter_grid: Dict[str, Sequence[Any]]


class ScoredModel(NamedTuple):
    pipeline: ModelPipeline
    parameters: Dict[str, Any]
    test_score_mean: float
    test_score_std: float
    ranking_score: float


class ModelRanker:
    """
    Turns a model zoo along with

        - a (optional) pre-processing pipeline
        - a cross-validation instance
        - a scoring function
    into a scikit-learn pipeline.

    :param models: list of model grids to be ranked
    :param cv: a cross validation object (i.e. CircularCrossValidator)
    :param scoring: a scorer to use when doing CV within GridSearch
    """

    __slots__ = ["_models", "_scoring", "_cv", "_searchers", "_pipeline"]

    F_PARAMETERS = "params"
    F_MEAN_TEST_SCORE = "mean_test_score"
    F_SD_TEST_SCORE = "std_test_score"

    def __init__(
        self,
        models: Iterable[Model],
        cv: Union[BaseCrossValidator, BaseShuffleSplit] = None,
        scoring=None,
    ) -> None:
        self._models = list(models)
        self._cv = cv
        self._scoring = scoring

    @staticmethod
    def default_ranking_scorer(mean_test_score: float, std_test_score: float) -> float:
        """
        The default scoring function to evaluate on top of GridSearchCV test scores,
        given by :code:`GridSearchCV.cv_results_`.

        Its output is used for ranking globally across the model zoo.

        :param mean_test_score: the mean test score across all folds for a (estimator,\
        parameters) combination
        :param std_test_score: the standard deviation of test scores across all folds \
        for a (estimator, parameters) combination

        :return: final score for a (estimator, parameters) combination
        """
        return mean_test_score - 2 * std_test_score

    def run(
        self,
        sample: Sample,
        ranking_scorer: Callable[[float, float], float] = None,
        n_jobs: Optional[int] = None,
        pre_dispatch="2*n_jobs",
    ) -> "ModelRanking":
        """
        Execute the pipeline with the given sample and return the ranking.

        :param sample: sample to fit pipeline to
        :param ranking_scorer: scoring function used for ranking across models,
        taking mean and standard deviation of the ranking scores and returning the
        overall ranking score (default: ModelRanking.default_ranking_scorer)
        :param n_jobs: number of threads to use (default: one)
        :param pre_dispatch: maximum number of the data to make (default: 2*n_jobs)

        :return the created model ranking of type :code:`ModelRanking`

        """

        # construct searchers
        searchers: List[Tuple[GridSearchCV, Model]] = [
            (
                GridSearchCV(
                    estimator=model.pipeline.pipeline,
                    cv=self._cv,
                    param_grid=model.parameter_grid,
                    scoring=self._scoring,
                    return_train_score=False,
                    n_jobs=n_jobs,
                    pre_dispatch=pre_dispatch,
                    refit=False,
                ),
                model,
            )
            for model in self._models
        ]

        for searcher, _ in searchers:
            searcher.fit(X=sample.features, y=sample.target)

        if ranking_scorer is None:
            ranking_scorer = ModelRanker.default_ranking_scorer

        # consolidate results of all searchers into "results"
        scored_models = [
            ScoredModel(
                pipeline=model.pipeline,
                parameters=params,
                test_score_mean=test_score_mean,
                test_score_std=test_score_std,
                # compute the final score using function defined above:
                ranking_score=ranking_scorer(test_score_mean, test_score_std),
            )
            for search, model in searchers
            # we read and iterate over these 3 attributes from cv_results_:
            for params, test_score_mean, test_score_std in zip(
                search.cv_results_[ModelRanker.F_PARAMETERS],
                search.cv_results_[ModelRanker.F_MEAN_TEST_SCORE],
                search.cv_results_[ModelRanker.F_SD_TEST_SCORE],
            )
        ]

        # create ranking by assigning rank values and creating "RankedModel" types
        return ModelRanking(
            ranking=sorted(
                scored_models, key=lambda model: model.ranking_score, reverse=True
            )
        )


class ModelRanking:
    BEST_MODEL_RANK = 0

    """
    Utility class that wraps a list of RankedModel
    """

    def __init__(self, ranking: List[ScoredModel]):
        """
        Utility class that wraps a list of RankedModel

        :param ranking: the list of RankedModel instances this ranking is based on
        """
        self.__ranking = ranking

    def model(self, rank: int) -> ScoredModel:
        """
        Returns the model instance at a given rank.

        :param rank: the rank of the model to get
        
        :return: a RankedModel instance
        """
        return self.__ranking[rank]

    def summary_report(self, limit: int = 10) -> str:
        """
        Generates a summary string of the best model instances

        :param limit: How many ranks to max. output

        :return: str
        """

        ranking = self.__ranking[:limit]

        def _model_name(ranked_model: ScoredModel) -> str:
            return ranked_model.pipeline.estimator.__class__.__name__

        name_width = max([len(_model_name(ranked_model)) for ranked_model in ranking])

        def parameters(params: Dict[str, Iterable[Any]]) -> str:
            return "\n    ".join(
                [
                    f"{param_name}={param_value}"
                    for param_name, param_value in params.items()
                ]
            )

        return "\n".join(
            [
                f" Rank {rank + 1:2d}: "
                f"{_model_name(ranked_model):>{name_width}s}, "
                f"Score={ranked_model.ranking_score:+.2e}, "
                f"Test mean={ranked_model.test_score_mean:+.2e}, "
                f"Test std={ranked_model.test_score_std:+.2e}"
                f"\n    {parameters(ranked_model.parameters)}"
                for rank, ranked_model in enumerate(ranking)
            ]
        )

    def __len__(self) -> int:
        return len(self.__ranking)

    def __str__(self) -> str:
        return self.summary_report()

    def __iter__(self) -> Iterable[ScoredModel]:
        return iter(self.__ranking)
