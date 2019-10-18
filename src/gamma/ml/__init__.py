"""
The Gamma machine learning library
"""

from copy import copy
from typing import *

import pandas as pd

from gamma.common import is_list_like, ListLike


class Sample:
    """
    Manages named features and target variables from a set of observations.

    Also provides basic methods for conveniently selecting subsets of features
    or observations.
    """

    __slots__ = ["_observations", "_target", "_features"]

    def __init__(
        self,
        observations: pd.DataFrame,
        target: Union[str, ListLike[str]],
        features: ListLike[str] = None,
    ) -> None:
        """
        Construct a Sample object.

        :param observations: a Pandas DataFrame
        :param target: string or list-like of strings naming the columns that \
            represent the target variable(s)
        :param features: optional list-like of strings naming the columns that \
            represent feature variables; or ``None`` (default), in which case all \
            non-target columns are considered to be features
        """

        def _ensure_columns_exist(column_type: str, columns: Iterable[str]):
            # check if all provided feature names actually exist in the observations df
            missing_columns = [
                name for name in columns if not observations.columns.contains(key=name)
            ]
            if len(missing_columns) > 0:
                missing_columns_list = '", "'.join(missing_columns)
                raise KeyError(
                    f"observations table is missing {column_type} columns "
                    f'{", ".join(missing_columns_list)}'
                )

        if observations is None or not isinstance(observations, pd.DataFrame):
            raise ValueError("sample is not a DataFrame")

        self._observations = observations

        multi_target = is_list_like(target)

        if multi_target:
            _ensure_columns_exist(column_type="target", columns=target)
        else:
            if target not in self._observations.columns:
                raise KeyError(
                    f'target "{target}" is not a column in the observations table'
                )

        self._target = target

        if features is None:
            features = observations.columns.drop(labels=target)
        else:
            _ensure_columns_exist(column_type="feature", columns=features)

            # ensure features and target(s) do not overlap

            if multi_target:
                shared = set(target).intersection(features)
                if len(shared) > 0:
                    raise KeyError(
                        f'targets {", ".join(shared)} are also included in the features'
                    )
            else:
                if target in features:
                    raise KeyError(f"target {target} is also included in the features")

        self._features = features

    @property
    def index(self) -> pd.Index:
        """Index of all observations in this sample."""
        return self.target.index

    @property
    def target(self) -> Union[pd.Series, pd.DataFrame]:
        """
        :return: the target as a pandas Series (if the target is a single column), or \
            as a pandas DataFrame if the Sample has multiple target columns
        """
        return self._observations.loc[:, self._target]

    @property
    def features(self) -> pd.DataFrame:
        """
        :return: all feature columns as a data frame
        """
        return self._observations.loc[:, self._features]

    def subsample(
        self,
        *,
        loc: Optional[Union[slice, ListLike[Any]]] = None,
        iloc: Optional[Union[slice, ListLike[int]]] = None,
    ) -> "Sample":
        """
        Select observations either by indices (`loc` parameter), or integer indices
        (`iloc` parameter). Exactly one of both parameters must be provided when
        calling this method, not both.

        :param loc: indices of observations to select
        :param iloc: integer indices of observations to select
        :return: copy of this sample, comprising only the observations at the given \
            index locations
        """
        subsample = copy(self)
        if iloc is None:
            if loc is None:
                ValueError("either arg loc or arg iloc must be specified")
            else:
                subsample._observations = self._observations.loc[loc, :]
        elif loc is None:
            subsample._observations = self._observations.iloc[iloc, :]
        else:
            raise ValueError(
                "arg loc and arg iloc must not both be specified at the same time"
            )
        return subsample

    def select_features(self, features: ListLike[str]) -> "Sample":
        """
        Return a Sample object which only includes the given features

        :param features: names of features to be selected
        :return: copy of this sample, containing only the features with the given names
        """
        if not set(features).issubset(self._features):
            raise ValueError(
                "arg features is not a subset of the features in this sample"
            )

        subsample = copy(self)
        subsample._features = features

        target = self._target
        if not is_list_like(target):
            target = [target]

        subsample._observations = self._observations.loc[:, [*features, *target]]

        return subsample

    def replace_features(self, features: pd.DataFrame) -> "Sample":
        """
        Return a new sample with this sample's target vector, and features replaced
        with the given features data frame. The index if the given features must be
        compatible with the index of this sample's observations.
        :param features: the features to replace the current features with
        :return: new Sample object with the replaced features
        """
        target = self.target

        if not features.index.isin(self.index).all():
            raise ValueError(
                "index of arg features contains items that do not exist in this sample"
            )

        return Sample(
            observations=features.join(target),
            target=target.name if isinstance(target, pd.Series) else target.columns,
        )

    def __len__(self) -> int:
        """
        :return: the number of observations in this sample
        """
        return len(self._observations)
