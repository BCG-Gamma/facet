"""
Simulation drawer.

:class:`SimulationDrawer` draws a simulation plot with on the x axis the feature
values in the simulation and on the y axis the associated prediction uplift. Below
this graph there is a histogram of the feature values.
"""

from typing import *

from facet.simulation import UnivariateSimulation
from facet.simulation.partition import T_Number
from pytools.viz import Drawer
from ._style import SimulationMatplotStyle, SimulationReportStyle, SimulationStyle


class _SimulationSeries(NamedTuple):
    # A set of aligned series representing the simulation result
    median_uplift: Sequence[T_Number]
    min_uplift: Sequence[T_Number]
    max_uplift: Sequence[T_Number]
    partitions: Sequence[T_Number]
    frequencies: Sequence[T_Number]


class SimulationDrawer(Drawer[UnivariateSimulation, SimulationStyle]):
    """
    Simulation drawer with high/low confidence intervals.
    """

    _STYLES = {"matplot": SimulationMatplotStyle, "text": SimulationReportStyle}

    def __init__(
        self, style: Union[SimulationStyle, str] = "matplot", histogram: bool = True
    ) -> None:
        """
        :param style: the style of the dendrogram; either as a \
            :class:`.SimulationStyle` instance, or as the name of a \
            default style. Permissible names are ``"matplot"`` for a style supporting \
            Matplotlib, and ``"text"`` for a text-only report to stdout \
            (default: ``"matplot"``)
        :param histogram: if ``True``, plot the histogram of observed values for the \
            feature being simulated; if ``False`` do not plot the histogram (default: \
            ``True``).
        """
        super().__init__(style=style)
        self._histogram = histogram

    def draw(self, data: UnivariateSimulation, title: Optional[str] = None) -> None:
        """
        Draw the simulation chart.
        :param data: the univariate simulation to draw
        :param title: the title of the chart (optional, defaults to a title \
            stating the name of the simulated feature)
        """
        if title is None:
            title = f"Simulation: {data.feature}"
        super().draw(data=data, title=title)

    @classmethod
    def _get_style_dict(cls) -> Mapping[str, Type[SimulationStyle]]:
        return SimulationDrawer._STYLES

    def _draw(self, data: UnivariateSimulation) -> None:
        # draw the simulation chart
        simulation_series = self._get_simulation_series(simulation=data)

        # draw the graph with the uplift curves
        self._style.draw_uplift(
            feature=data.feature,
            target=data.target,
            min_percentile=data.min_percentile,
            max_percentile=data.max_percentile,
            is_categorical_feature=data.partitioner.is_categorical,
            partitions=simulation_series.partitions,
            frequencies=simulation_series.frequencies,
            median_uplift=simulation_series.median_uplift,
            min_uplift=simulation_series.min_uplift,
            max_uplift=simulation_series.max_uplift,
        )

        if self._histogram:
            # draw the histogram of the simulation values
            self._style.draw_histogram(
                partitions=simulation_series.partitions,
                frequencies=simulation_series.frequencies,
                is_categorical_feature=data.partitioner.is_categorical,
            )

    @staticmethod
    def _get_simulation_series(simulation: UnivariateSimulation) -> _SimulationSeries:
        # return the simulation series for median uplift, min uplift, max uplift,
        # partitions and frequencies
        # If the partitioning of the simulation is categorical, the series are
        # sorted in ascending order of the median uplift.
        # Otherwise, the simulation series are returned unchanged.

        simulation_series = _SimulationSeries(
            simulation.median_change,
            simulation.min_change,
            simulation.max_change,
            simulation.partitioner.partitions(),
            simulation.partitioner.frequencies(),
        )

        if simulation.partitioner.is_categorical:
            # for categorical features, sort the categories by the median uplift
            return _SimulationSeries(
                *zip(*sorted(zip(*simulation_series), key=lambda x: x[0]))
            )
        else:
            return simulation_series
