import pandas as pd
import numpy as np
from typing import Generator, Tuple


class DataSplitter:
    """
    Class to generate various CV folds of train and test datasets using circular bootstrapping.

    Compatible with scikit-learn's GridSearchCV object, if you set :code:`cv=datasplitter.get_train_test_splits_as_indices()`

    See scikit-learn's `code <https://github.com/scikit-learn/scikit-learn/blob/7b136e9/sklearn/model_selection/_search.py#L961>`_
    and `reference <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html>`_

    Unless you call :code:`resample()`, **this class will keep drawn folds stable**

    todo: Evaluate if this should implement https://scikit-learn.org/stable/glossary.html#term-cv-splitter

    :param num_samples: Number of samples of the input dataset - needed to give deterministic folds on multiple\
     call of get_train_test_splits_as_indices
    :param test_ratio:  Ratio determining the size of the test set (default=0.2).
    :param num_folds:   Number of folds to generate (default=50).
    """

    def __init__(
        self, num_samples: int = None, test_ratio: float = 0.2, num_folds: int = 50
    ):
        """
        :param num_samples: Number of samples of the input dataset - needed to give deterministic folds on multiple\
         call of get_train_test_splits_as_indices
        :param test_ratio:  Ratio determining the size of the test set (default=0.2).
        :param num_folds:   Number of folds to generate (default=50).
        """

        self.__test_ratio = test_ratio
        self.__num_folds = num_folds
        self.__num_samples = num_samples

        if num_samples is None or not type(num_samples) == int or not num_samples > 0:
            raise ValueError("num_samples needs to be specified, of type int and > 0")

        if not (0 < self.__test_ratio < 1):
            raise ValueError(
                "Expected (0 < test_ratio < 1), but %d was given" % test_ratio
            )

        self.__num_test_samples_per_fold = int(self.__num_samples * self.__test_ratio)

        if self.__num_test_samples_per_fold == 0:
            raise ValueError(
                "The number of test samples per fold is 0 - increase ratio or size of input dataset"
            )

        self.__test_splits_start_samples = np.random.randint(
            0, self.__num_samples - 1, self.__num_folds
        )
        self.__data_indices = np.arange(self.__num_samples)

    def resample(self):
        """
        Draws completely new random start samples and will hence yield completely new folds from then on
        :return:
        """
        self.__test_splits_start_samples = np.random.randint(
            0, self.__num_samples - 1, self.__num_folds
        )

    def get_train_test_splits_as_indices(self):
        """
        Retrieves all generated folds of (train, test) pairs as tuples of arrays with the indices

        Meant to be used as "cv" parameter inside scikit-learn's GridSearchCV

        :return: A generator of tuples of kind (ndarray, ndarray). If you need a list, simply \
        call :code:`list(datasplitter.get_train_test_splits_as_indices(...))`
        """
        for fold_test_start_sample in self.__test_splits_start_samples:
            data_indices_rolled = np.roll(self.__data_indices, fold_test_start_sample)
            test_indices = data_indices_rolled[0 : self.__num_test_samples_per_fold]
            train_indices = data_indices_rolled[self.__num_test_samples_per_fold :]
            # conform to scikit-learn, expecting " - An iterable yielding (train, test) splits as arrays of indices."
            # see: https://github.com/scikit-learn/scikit-learn/blob/7b136e9/sklearn/model_selection/_search.py#L961
            yield (train_indices, test_indices)

    def get_train_test_splits_as_dataframes(
        self, input_dataset: pd.DataFrame
    ) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
        """
        Retrieves all generated folds of (train, test) pairs as tuples of dataframes

        :param input_dataset: A pd.DataFrame object containing all data to split.
        :return: A generator of tuples of kind (pd.DataFrame, pd.DataFrame). If you need a list, simply \
        call :code:`list(datasplitter.get_train_test_splits_as_dataframes(...))`
        """
        if not type(input_dataset) == pd.DataFrame:
            raise ValueError("Expected a pandas.DataFrame as input_dataset")

        for (train_indices, test_indices) in self.get_train_test_splits_as_indices():
            yield (
                (input_dataset.iloc[train_indices], input_dataset.iloc[test_indices])
            )
