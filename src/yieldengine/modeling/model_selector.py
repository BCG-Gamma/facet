from sklearn.base import TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from typing import List
import pandas as pd

# note: unfortunately, sklearn does not expose "BaseSearchCV" from within model_selection, which is the superclass


class ModelSelector:
    """
    Class that helps in training, validating and ranking multiple scikit-learn models
    (i.e. various regressor implementations), that depend all on the same (or none) pre-processing pipeline

    For an example, please see:
    `Example: Loading, Preprocessing, Circular CV, Model Selection  <./examples/e1.html>`_
    """

    def __init__(
        self, searchers: List[GridSearchCV], preprocessing: Pipeline = None
    ) -> None:
        """
        Constructs a ModelSelector

        :param searchers: a list of scikit-learn GridSearchCV objects (not fitted)
        :param preprocessing: a scikit-learn Pipeline that should be used as a preprocessor
        :return None
        """
        self.__searchers = searchers
        self.__preprocessing = preprocessing
        self.__pipeline = None

    def construct_pipeline(self) -> Pipeline:
        """
        Constructs and returns a single scikit-learn Pipeline, comprising of all given searchers and using the
        (if supplied) preprocessing step.

        All given :code:`searchers` will be included into the pipeline as a parallel step leveraging
        :code:`FeatureUnion`. This means, calling :code:`fit(X, y)` on it will run all gridsearchers, which you then
        can inspect using :code:`model_selector.rank_models()`, :code:`model_selector.rank_model_instances()` etc.

        Running :code:`transform(X)` on the pipeline will yield a (#samples X #searchers) shaped numpy array
        with predictions that each model made.

        :return: Pipeline
        """

        class ModelTransformer(TransformerMixin):
            def __init__(self, model):
                self.model = model

            def fit(self, *args, **kwargs):
                self.model.fit(*args, **kwargs)
                return self

            def transform(self, X, **transform_params):
                return pd.DataFrame(self.model.predict(X))

        # generate a list of (name, obj) tuples for all given estimators (=GridSearchCV objs.):
        #   name: simply e0, e1, e2, ...
        #   obj: the GridSearchCV wrapped in a ModelTransformer() object, to be feature union compliant
        estimator_steps = [
            (f"e{i}", ModelTransformer(estimator))
            for i, estimator in enumerate(self.__searchers)
        ]

        # with the above created list of ModelTransformers, create FeatureUnion()
        # benefit: can be evaluated in parallel and pre-processing is shared
        est_feature_union = (
            "estimators",
            FeatureUnion(transformer_list=estimator_steps),
        )

        # if pre-processing pipeline was given, insert it into the pipeline:
        if self.__preprocessing is not None:
            self.__pipeline = Pipeline(
                [("preprocessing", self.__preprocessing), est_feature_union]
            )
        else:
            # if pre-processing pipeline was not given, create a minimal Pipeline with just the estimators
            self.__pipeline = Pipeline([est_feature_union])

        return self.__pipeline

    def rank_models(self) -> List[GridSearchCV]:
        """
        Ranks models associated with this ModelSelector - i.e. the searchers.

        This is done by looking at :code:`best_score_` of each searcher and sorting by this value in descending fashion.

        **Note:** In case you have defined your own scorer for a searcher,
        ensure you have defined :code:`greater_is_better` correctly.

        :return: List[GridSearchCV] (in descending order from best score to worst score)
        """
        l = self.__searchers.copy()

        # always sort best_score_ in descending fashion, since gridsearchcv flips the sign of its score value when
        # the applied scoring method has defined :code:greater_is_better=False)
        l.sort(key=lambda x: x.best_score_ * -1)
        return l

    def rank_model_instances(self, n_best_ranked=3) -> List[dict]:
        """
        Ranks model instances associated with this ModelSelector - i.e. individual (estimator, parameter) pairs

        This means we look at all searchers, get the :code:`n_best_ranked` model instances for each of them (including ties)
        and then finally do a complete rank using their mean_test_score.

        A list of dictionaries with keys (estimator, score, params) is returned. If you need a DataFrame, simply do a
        pd.DataFrame(..) on it.

        :param n_best_ranked: Defines how many best model instances to return per individual model we grid serched. \
        This looks at the :code:`rank_test_score` value of scikit-learn's :code:`GridSearchCV.cv_results_` object. \
        It might retrieve tied pairs of (estimator, parameters) for completeness.

        :return: List[dict]

        """
        ranked_model_instances = []

        for s in self.__searchers:
            all_model_instances = pd.DataFrame(s.cv_results_)
            best_model_instances = all_model_instances[
                all_model_instances["rank_test_score"] <= n_best_ranked
            ]

            best_model_instances_list = best_model_instances[
                ["mean_test_score", "params"]
            ].values.tolist()

            results_for_s = [
                {"estimator": s.estimator, "score": i[0], "params": i[1]}
                for i in best_model_instances_list
            ]

            ranked_model_instances.extend(results_for_s)

        ranked_model_instances.sort(key=lambda x: x["score"] * -1)

        return ranked_model_instances

    def summary_string(self, limit: int = 25) -> str:
        """
        Generates a summary string of the best models, ranked by :code:`rank_modls()`

        :param limit: How many ranks to max. output

        :return: str
        """
        return "\n".join(
            [
                f" Rank {i + 1}: {m.estimator.__class__}, Score: {-1 * m.best_score_}, Params: {m.best_params_}"
                for i, m in enumerate(self.rank_models())
                if i < limit
            ]
        )
