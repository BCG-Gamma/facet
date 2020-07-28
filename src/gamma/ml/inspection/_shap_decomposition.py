"""
Decomposition of SHAP contribution scores (i.e, SHAP importance) of all possible parings
of features into additive components for synergy, redundancy, and independence.
"""
import logging
from typing import *

import numpy as np
import pandas as pd

from gamma.common.fit import FittableMixin, T_Self
from gamma.ml.inspection._shap import ShapCalculator, ShapInteractionValuesCalculator

log = logging.getLogger(__name__)

_PAIRWISE_PARTIAL_SUMMATION = False
#: if `True`, optimize numpy arrays to ensure pairwise partial summation.
#: But given that we will add floats of the same order of magnitude and only up
#: to a few thousand of them in the base case, the loss of accuracy with regular
#: (sequential) summation will be negligible in practice


class ShapValueDecomposer(FittableMixin[ShapCalculator]):
    """
    Decomposes SHAP vectors (i.e., SHAP contribution) of all possible parings
    of features into additive components for association and independence.
    SHAP contribution scores are calculated as the standard deviation of the individual
    interactions per observation. Using this metric, rather than the mean of absolute
    interactions, allows us to calculate the decomposition without ever constructing
    the decompositions of the actual SHAP vectors across observations.
    """

    def __init__(self) -> None:
        super().__init__()
        self.index_: Optional[pd.Index] = None
        self.columns_: Optional[pd.Index] = None
        self.association_rel_: Optional[np.ndarray] = None

    @property
    def is_fitted(self) -> bool:
        """[inherit docstring from parent class]"""
        return self.index_ is not None

    is_fitted.__doc__ = FittableMixin.is_fitted.__doc__

    def fit(self: T_Self, shap_calculator: ShapCalculator, **fit_params) -> T_Self:
        """
        Calculate the SHAP decomposition for the shap values produced by the
        given SHAP calculator.
        :param shap_calculator: the fitted calculator from which to get the shap values
        """

        self: ShapValueDecomposer  # support type hinting in PyCharm

        successful = False
        try:
            if len(fit_params) > 0:
                raise ValueError(
                    f'unsupported fit parameters: {", ".join(fit_params.values())}'
                )

            self._fit(shap_calculator=shap_calculator)

            self.index_ = shap_calculator.feature_index_
            self.columns_ = shap_calculator.shap_columns

        except Exception:
            # reset fit in case we get an exception along the way
            self._reset_fit()
            raise

        return self

    @property
    def association(self) -> pd.DataFrame:
        """
        The matrix of relative association for all feature pairs.

        Values range between 0.0 (fully independent contributions) and 1.0
        (fully associated contributions).

        Raises an error if this SHAP value decomposer has not been fitted.
        """
        self._ensure_fitted()
        return self._to_frame(self.association_rel_)

    def _fit(self: T_Self, shap_calculator: ShapCalculator) -> None:
        #
        # basic definitions
        #

        shap_values = shap_calculator.get_shap_values(consolidate=None)
        n_targets = len(shap_calculator.target_columns_)
        n_features = len(shap_calculator.feature_index_)
        n_observations = len(shap_values)

        # p[i] = p_i
        # shape: (n_targets, n_features, n_observations)
        # the vector of shap values for every target and feature
        p_i = _ensure_last_axis_is_fast(
            np.transpose(
                shap_values.values.reshape((n_observations, n_targets, n_features)),
                axes=(1, 2, 0),
            )
        )

        #
        # SHAP association: ass[i, j]
        #

        # covariance matrix of shap vectors
        # shape: (n_targets, n_features, n_features)
        cov_p_i_p_j = _cov(p_i, p_i)
        cov_p_i_p_j_2x = 2 * cov_p_i_p_j

        # variance of shap vectors
        var_p_i = np.diagonal(cov_p_i_p_j, axis1=1, axis2=2)
        var_p_i_plus_var_p_j = var_p_i[:, :, np.newaxis] + var_p_i[:, np.newaxis, :]

        # std(p[i] + p[j])
        # shape: (n_targets, n_features)
        # variances of SHAP vectors minus total synergy
        # or, the length of the sum of vectors p[i] and p[j]
        # this quantifies the joint contributions of features i and j
        # we also need this as part of the formula to calculate ass_ij (see below)
        std_p_i_plus_p_j = _sqrt(var_p_i_plus_var_p_j + cov_p_i_p_j_2x)

        # std(p[i] - p[j])
        # shape: (n_targets, n_features)
        # the length of the difference of vectors p[i] and p[j]
        # we need this as part of the formula to calculate ass_ij (see below)
        std_p_i_minus_p_j = _sqrt(var_p_i_plus_var_p_j - cov_p_i_p_j_2x)

        # 2 * std(ass[i, j]) = 2 * std(ass[j, i])
        # shape: (n_targets, n_features, n_features)
        # twice the standard deviation (= length) of the redundancy vector;
        # this is the total contribution made by p[i] and p[j] where both features
        # independently use redundant information
        std_ass_ij_2x = std_p_i_plus_p_j - std_p_i_minus_p_j

        # SHAP association
        # shape: (n_targets, n_features, n_features)
        association_ij = np.abs(std_ass_ij_2x)

        # we should have the right shape for all resulting matrices
        assert association_ij.shape == (n_targets, n_features, n_features)

        with np.errstate(divide="ignore", invalid="ignore"):
            self.association_rel_ = _ensure_diagonality(
                association_ij / std_p_i_plus_p_j
            )

    def _reset_fit(self) -> None:
        # revert status of this object to not fitted
        self.index_ = self.columns_ = None
        self.association_rel_ = None

    def _to_frame(self, matrix: np.ndarray) -> pd.DataFrame:
        # takes an array of shape (n_targets, n_features, n_features) and transforms it
        # into a data frame of shape (n_features, n_targets * n_features)
        index = self.index_
        columns = self.columns_
        return pd.DataFrame(
            matrix.swapaxes(0, 1).reshape(len(index), len(columns)),
            index=index,
            columns=columns,
        )


class ShapInteractionValueDecomposer(ShapValueDecomposer):
    """
    Decomposes SHAP interaction scores (i.e, SHAP importance) of all possible parings
    of features into additive components for synergy, redundancy, and independence.
    SHAP interaction scores are calculated as the standard deviation of the individual
    interactions per observation. Using this metric, rather than the mean of absolute
    interactions, allows us to calculate the decomposition without ever constructing
    the decompositions of the actual SHAP vectors across observations.
    """

    DEFAULT_MIN_DIRECT_SYNERGY = 0.01

    def __init__(self, min_direct_synergy: Optional[float] = None) -> None:
        """
        :param min_direct_synergy: minimum direct synergy a pair of features \
            :math:`f_i' and :math:`f_j' needs to manifest in order to be considered \
            for calculating indirect synergies. This is expressed as the relative \
            contribution score with regard to the total synergistic contributions \
            ranging between 0 and 1, and calculated as \
            :math:`\\frac \
                    {\\sigma_{\\vec{\\phi_{ij}}}} \
                    {\\sum_{i,j}\\sigma_{\\vec{\\phi_{ij}}}}`, \
            i.e, the relative share of the synergy contribution \
            :math:`\\sigma_{\\vec{\\phi_{ij}}}`. \
        """
        super().__init__()
        self.min_direct_synergy = (
            ShapInteractionValueDecomposer.DEFAULT_MIN_DIRECT_SYNERGY
            if min_direct_synergy is None
            else min_direct_synergy
        )
        self.synergy_rel_: Optional[np.ndarray] = None
        self.redundancy_rel_: Optional[np.ndarray] = None

    __init__.__doc__ += f"""\
            (default: {DEFAULT_MIN_DIRECT_SYNERGY}, i.e., \
            {DEFAULT_MIN_DIRECT_SYNERGY * 100.0:g}%)
        """

    def fit(
        self: T_Self, shap_calculator: ShapInteractionValuesCalculator, **fit_params
    ) -> T_Self:
        """
        Calculate the SHAP decomposition for the shap values produced by the
        given SHAP interaction values calculator.
        :param shap_calculator: the fitted calculator from which to get the shap values
        """
        return super().fit(shap_calculator=shap_calculator, **fit_params)

    @property
    def synergy(self) -> pd.DataFrame:
        """
        The matrix of total relative synergy (direct and indirect) for all feature
        pairs.

        Values range between 0.0 (fully autonomous contributions) and 1.0
        (fully synergistic contributions).

        Raises an error if this interaction decomposer has not been fitted.
        """
        self._ensure_fitted()
        return self._to_frame(self.synergy_rel_)

    @property
    def redundancy(self) -> pd.DataFrame:
        """
        The matrix of total relative redundancy for all feature pairs.

        Values range between 0.0 (fully unique contributions) and 1.0
        (fully redundant contributions).

        Raises an error if this interaction decomposer has not been fitted.
        """
        self._ensure_fitted()
        return self._to_frame(self.redundancy_rel_)

    def _fit(self, shap_calculator: ShapInteractionValuesCalculator) -> None:
        super()._fit(shap_calculator)

        #
        # basic definitions
        #

        shap_values = shap_calculator.get_shap_interaction_values(consolidate=None)
        features = shap_calculator.feature_index_
        targets = shap_calculator.target_columns_
        n_features = len(features)
        n_targets = len(targets)
        n_observations = shap_values.shape[0] // n_features

        # p[i, j]
        # shape: (n_targets, n_features, n_features, n_observations)
        # the vector of interaction values for every target and feature pairing
        # for improved numerical precision, we ensure the last axis is the fast axis
        # i.e. stride size equals item size (see documentation for numpy.sum)
        p_ij = _ensure_last_axis_is_fast(
            np.transpose(
                shap_values.values.reshape(
                    (n_observations, n_features, n_targets, n_features)
                ),
                axes=(2, 1, 3, 0),
            )
        )

        # p[i]
        # shape: (n_targets, n_features, n_observations)
        p_i = _ensure_last_axis_is_fast(p_ij.sum(axis=2))

        # covariance matrix of shap vectors
        # shape: (n_targets, n_features, n_features)
        cov_p_i_p_j = _cov(p_i, p_i)

        #
        # Feature synergy (direct and indirect): zeta[i, j]
        #

        # var(p[i, j]), std(p[i, j])
        # shape: (n_targets, n_features, n_features)
        # variance and length (= standard deviation) of each feature interaction vector
        var_p_ij = _ensure_last_axis_is_fast(p_ij * p_ij).sum(axis=-1) / n_observations
        std_p_ij = np.sqrt(var_p_ij)

        # p[i, i]
        # shape: (n_targets, n_features, n_observations)
        # independent feature contributions;
        # this is the diagonal of p_ij
        p_ii = np.diagonal(p_ij, axis1=1, axis2=2).swapaxes(1, 2)

        # p'[i] = p[i] - p[i, i]
        # shape: (n_targets, n_features, n_observations)
        # the SHAP vectors per feature, minus the independent contributions
        p_prime_i = _ensure_last_axis_is_fast(p_i - p_ii)

        # std_p_relative[i, j]
        # shape: (n_targets, n_features, n_features)
        # relative importance of p[i, j] measured as the length of p[i, j]
        # as percentage of the sum of lengths of all p[..., ...]
        with np.errstate(divide="ignore", invalid="ignore"):
            std_p_relative_ij = std_p_ij / std_p_ij.sum()

        # p_valid[i, j]
        # shape: (n_targets, n_features, n_features)
        # boolean values indicating whether p[i, j] is above the "noise" threshold,
        # i.e. whether we trust that doing calculations with p[i, j] is sufficiently
        # accurate. p[i, j] with small variances should not be used because we divide
        # by that variance to determine the multiple for indirect synergy, which can
        # deviate significantly if var(p[i, j]) is only slightly off

        interaction_noise_threshold = self.min_direct_synergy
        p_valid_ij = std_p_relative_ij >= interaction_noise_threshold

        # k[i, j]
        # shape: (n_targets, n_features, n_features)
        # k[i, j] = cov(p'[i], p[i, j]) / var(p[i, j]), for each target
        # this is the orthogonal projection of p[i, j] onto p'[i] and determines
        # the multiplier of std(p[i, j]) (i.e., direct synergy) to obtain
        # the total direct and indirect synergy of p'[i] with p'[j], i.e.,
        # * k[i, j] * std(p[i, j])

        if _PAIRWISE_PARTIAL_SUMMATION:
            raise NotImplementedError(
                "max precision Einstein summation not yet implemented"
            )
        # noinspection SpellCheckingInspection
        k_ij = np.divide(
            # cov(p'[i], p[i, j])
            np.einsum("tio,tijo->tij", p_prime_i, p_ij, optimize=True) / n_observations,
            # var(p[i, j])
            var_p_ij,
            out=np.ones_like(var_p_ij),
            where=p_valid_ij,
        )

        # issue warning messages for edge cases

        def _feature(_i: int) -> str:
            return f'"{features[_i]}"'

        def _for_target(_t: int) -> str:
            if targets is None:
                return ""
            else:
                return f' for target "{targets[_t]}"'

        def _relative_direct_synergy(_t: int, _i: int, _j: int) -> str:
            return (
                f"p[{features[_i]}, {features[_j]}] has "
                f"{std_p_relative_ij[_t, _i, _j] * 100:.3g}% "
                "relative SHAP contribution; "
                "consider increasing the interaction noise threshold (currently "
                f"{interaction_noise_threshold * 100:.3g}%). "
            )

        def _test_synergy_feasibility() -> None:
            for _t, _i, _j in np.argwhere(k_ij < 1):
                if _i != _j:
                    log.debug(
                        "contravariant indirect synergy "
                        f"between {_feature(_i)} and {_feature(_j)}{_for_target(_t)}: "
                        "indirect synergy calculated as "
                        f"{(k_ij[_t, _i, _j] - 1) * 100:.3g}% "
                        "of direct synergy; setting indirect synergy to 0. "
                        f"{_relative_direct_synergy(_t, _i, _j)}"
                    )

            for _t, _i, _j in np.argwhere(k_ij - 1 > np.log2(n_features)):
                if _i != _j:
                    log.warning(
                        "high indirect synergy "
                        f"between {_feature(_i)} and {_feature(_j)}{_for_target(_t)}: "
                        "total synergy is "
                        f"{k_ij[_t, _i, _j] * 100:.3g}% of direct synergy. "
                        f"{_relative_direct_synergy(_t, _i, _j)}"
                    )

        _test_synergy_feasibility()

        # ensure that s[i, j] is at least 1.0
        # (a warning will have been issued during the checks above for s[i, j] < 1)
        # i.e. we don't allow total synergy to be less than direct synergy
        k_ij = np.clip(k_ij, 1.0, None)

        # fill the diagonal(s) with nan since these values are meaningless
        for k_ij_for_target in k_ij:
            np.fill_diagonal(k_ij_for_target, val=np.nan)

        # s[j, i]
        # transpose of s[i, j]; we need this later for calculating SHAP redundancy
        # shape: (n_targets, n_features, n_features)
        k_ji = _transpose(k_ij)

        # syn[i, j] = syn[j, i] = (k[i, j] + k[j, i]) * p[i, j]
        # total SHAP synergy, comprising both direct and indirect synergy
        # shape: (n_targets, n_features, n_features)
        std_syn_ij_plus_syn_ji = (k_ij + k_ji) * std_p_ij

        #
        # SHAP autonomy: aut[i, j]
        #

        # cov(p[i], p[i, j])
        # covariance matrix of shap vectors with pairwise synergies
        # shape: (n_targets, n_features, n_features)

        if _PAIRWISE_PARTIAL_SUMMATION:
            raise NotImplementedError(
                "max precision Einstein summation not yet implemented"
            )
        cov_p_i_p_ij = np.einsum("...io,...ijo->...ij", p_i, p_ij) / n_observations

        # cov(aut[i, j], aut[j, i])
        # where aut[i, j] = p[i] - k[i, j] * p[i, j]
        # shape: (n_observations, n_targets, n_features)
        # matrix of covariances for tau vectors for all pairings of features
        # the aut[i, j] vector is the p[i] SHAP contribution vector where the synergy
        # effects (direct and indirect) with feature j have been deducted

        cov_aut_ij_aut_ji = (
            cov_p_i_p_j
            - k_ji * cov_p_i_p_ij
            - k_ij * _transpose(cov_p_i_p_ij)
            + k_ji * k_ij * var_p_ij
        )

        # var(p[i])
        # variances of SHAP vectors
        # shape: (n_targets, n_features, 1)
        # i.e. adding a second, empty feature dimension to enable correct broadcasting
        var_p_i = np.diagonal(cov_p_i_p_j, axis1=1, axis2=2)[:, :, np.newaxis]

        # var(aut[i, j])
        # variances of SHAP vectors minus total synergy
        # shape: (n_targets, n_features, n_features)
        var_aut_ij = var_p_i - 2 * k_ij * cov_p_i_p_ij + k_ij * k_ij * var_p_ij

        # var(aut[i]) + var(aut[j])
        # Sum of covariances per feature pair (this is a diagonal matrix)
        # shape: (n_targets, n_features, n_features)
        var_aut_ij_plus_var_aut_ji = var_aut_ij + _transpose(var_aut_ij)

        # 2 * cov(aut[i, j], aut[j, i])
        # shape: (n_targets, n_features, n_features)
        # this is an intermediate result to calculate the standard deviation
        # of the redundancy vector (see next step below)
        cov_aut_ij_aut_ji_2x = 2 * cov_aut_ij_aut_ji

        # std(aut[i, j] + aut[j, i])
        # where aut[i, j] = p[i] - syn[i, j]
        # shape: (n_targets, n_features, n_features)
        # variances of SHAP vectors minus total synergy
        # or, the length of the sum of vectors aut[i, j] and aut[j, i]
        # this quantifies the autonomous contributions of features i and j, i.e.,
        # without synergizing
        # we also need this as part of the formula to calculate red_ij (see below)

        std_aut_ij_plus_aut_ji = _sqrt(
            var_aut_ij_plus_var_aut_ji + cov_aut_ij_aut_ji_2x
        )

        #
        # SHAP redundancy: red[i, j]
        #

        # std(aut[i, j] - aut[j, i])
        # shape: (n_targets, n_features, n_features)
        # the length of the difference of vectors aut[i, j] and aut[j, i]
        # we need this as part of the formula to calculate red_ij (see below)

        std_aut_ij_minus_aut_ji = _sqrt(
            var_aut_ij_plus_var_aut_ji - cov_aut_ij_aut_ji_2x
        )

        # 2 * std(red[i, j]) = 2 * std(red[j, i])
        # shape: (n_targets, n_features)
        # twice the standard deviation (= length) of redundancy vector;
        # this is the total contribution made by p[i] and p[j] where both features
        # independently use redundant information

        std_red_ij_2x = std_aut_ij_plus_aut_ji - std_aut_ij_minus_aut_ji

        #
        # SHAP independence: ind[i, j]
        #

        # 4 * var(red[i, j]), var(red[i, j])
        # shape: (n_targets, n_features, n_features)
        var_red_ij_4x = std_red_ij_2x * std_red_ij_2x

        # ratio of length of 2*e over length of (aut[i, j] + aut[j, i])
        # shape: (n_targets, n_features, n_features)
        # we need this for the next step
        with np.errstate(divide="ignore", invalid="ignore"):
            red_aut_ratio_2x = 1 - std_aut_ij_minus_aut_ji / std_aut_ij_plus_aut_ji

        # 2 * cov(aut[i, j], red[i, j])
        # shape: (n_targets, n_features, n_features)
        # we need this as part of the formula to calculate nu_i (see next step below)
        cov_aut_ij_red_ij_2x = red_aut_ratio_2x * (var_aut_ij + cov_aut_ij_aut_ji)

        # std(ind_ij)
        # where ind_ij = aut_ij + aut_ji - 2 * red_ij
        # shape: (n_targets, n_features, n_features)
        # the standard deviation (= length) of the independence vector

        std_ind_ij_plus_ind_ji = _sqrt(
            var_aut_ij
            + _transpose(var_aut_ij)
            + var_red_ij_4x
            + 2
            * (
                cov_aut_ij_aut_ji
                - cov_aut_ij_red_ij_2x
                - _transpose(cov_aut_ij_red_ij_2x)
            )
        )

        #
        # SHAP uniqueness: uni[i, j]
        #

        # 2 * cov(p[i], red[i, j])
        # shape: (n_targets, n_features, n_features)
        # intermediate result to calculate upsilon[i, j], see next step

        cov_p_i_red_ij_2x = red_aut_ratio_2x * (
            var_p_i + cov_p_i_p_j - (k_ij + k_ji) * cov_p_i_p_ij
        )

        # std(uni[i, j] + uni[j, i])
        # where uni[i, j] + + uni[j, i] = p[i] + p[j] - 2 * red[i, j]
        # shape: (n_targets, n_features, n_features)
        # this is the sum of complementary contributions of feature i and feature j,
        # i.e., deducting the redundant contributions

        std_uni_ij_plus_uni_ji = _sqrt(
            var_p_i
            + var_p_i.swapaxes(1, 2)
            + var_red_ij_4x
            + 2 * (cov_p_i_p_j - cov_p_i_red_ij_2x - _transpose(cov_p_i_red_ij_2x))
        )

        #
        # SHAP decompositon as relative contributions of
        # synergy, redundancy, and independence
        #

        synergy_ij = std_syn_ij_plus_syn_ji
        autonomy_ij = std_aut_ij_plus_aut_ji
        redundancy_ij = np.abs(std_red_ij_2x)
        uniqueness_ij = std_uni_ij_plus_uni_ji
        independence_ij = std_ind_ij_plus_ind_ji

        # we should have the right shape for all resulting matrices
        for matrix in (
            synergy_ij,
            redundancy_ij,
            autonomy_ij,
            uniqueness_ij,
            independence_ij,
        ):
            assert matrix.shape == (n_targets, n_features, n_features)

        # calculate relative synergy and redundancy (ranging from 0.0 to 1.0)
        # both matrices are symmetric, but we ensure perfect symmetry by removing
        # potential round-off errors
        # NOTE: we do not store independence so technically it could be removed from
        # the code above

        with np.errstate(divide="ignore", invalid="ignore"):
            synergy_rel = _ensure_diagonality(synergy_ij / (synergy_ij + autonomy_ij))
            redundancy_rel = _ensure_diagonality(
                redundancy_ij / (redundancy_ij + uniqueness_ij)
            )

        self.synergy_rel_ = synergy_rel
        self.redundancy_rel_ = redundancy_rel

    def _reset_fit(self) -> None:
        # revert status of this object to not fitted
        super()._reset_fit()
        self.synergy_rel_ = self.redundancy_rel_ = None


def _ensure_diagonality(matrix: np.ndarray) -> np.ndarray:
    # matrix shape: (n_targets, n_features, n_features)

    # remove potential floating point round-off errors
    matrix = (matrix + _transpose(matrix)) / 2

    # fixes per target
    for m in matrix:
        # replace nan values with 0.0 = no association when correlation is undefined
        np.nan_to_num(m, copy=False)

        # set the matrix diagonals to 1.0 = full association of each feature with
        # itself
        np.fill_diagonal(m, 1.0)

    return matrix


def _ensure_last_axis_is_fast(v: np.ndarray) -> np.ndarray:
    if _PAIRWISE_PARTIAL_SUMMATION:
        if v.strides[-1] != v.itemsize:
            v = v.copy()
        assert v.strides[-1] == v.itemsize
    return v


def _cov(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    # calculate covariance matrix of two vectors, assuming µ=0 for both
    # input shape for u and v is (n_targets, n_features, n_observations)
    # output shape is (n_targets, n_features, n_features, n_observations)

    assert u.shape == v.shape
    assert u.ndim == 3

    if _PAIRWISE_PARTIAL_SUMMATION:
        raise NotImplementedError("max precision matmul not yet implemented")
    else:
        return np.matmul(u, v.swapaxes(1, 2)) / u.shape[2]


def _transpose(m: np.ndarray) -> np.ndarray:
    # transpose a feature matrix for all targets
    assert m.ndim == 3

    return m.swapaxes(1, 2)


def _sqrt(v: np.ndarray) -> np.ndarray:
    # we clip values < 0 as these could happen in isolated cases due to
    # rounding errors

    return np.sqrt(np.clip(v, 0, None))
