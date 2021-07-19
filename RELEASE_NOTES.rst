Release Notes
=============

FACET 1.2
---------

FACET 1.2 adds support for *sklearndf* 1.2 and *scikit-learn* 0.24.
It also introduces the ability to run simulations on a subsample of the data used to
fit the underlying crossfit.
One example where this can be useful is to use only a recent period of a time series as
the baseline of a simulation.

1.2.0
~~~~~

- BUILD: added support for *sklearndf* 1.2 and *scikit-learn* 0.24
- API: new optional parameter `subsample` in method 
  :meth:`.BaseUnivariateSimulator.simulate_feature` can be used to specify a subsample
  to be used in the simulation (but simulating using a crossfit based on the full
  sample)


FACET 1.1
---------

FACET 1.1 refines and enhances the association/synergy/redundancy calculations provided
by the :class:`.LearnerInspector`.

1.1.0
~~~~~

- API: SHAP interaction vectors can (in part) also be influenced by redundancy among
  features. This can inflate quantificatios of synergy, especially in cases where two
  variables are highly redundant. FACET now corrects interaction vectors for redundancy
  prior to calculating synergy. Technically we ensure that each interaction vector is
  orthogonal w.r.t the main effect vectors of both associated features.
- API: FACET now calculates synergy, redundancy, and association separately for each
  model in a crossfit, then returns the mean of all resulting matrices. This leads to a
  slight increase in accuracy, and also allows us to calculate the standard deviation
  across matrices as an indication of confidence for each calculated value.
- API: Method :meth:`.LernerInspector.shap_plot_data` now returns SHAP values for the
  positive class of binary classifiers.
- API: Increase efficiency of :class:`.LearnerRanker` parallelization by adopting the
  new :class:`pytools.parallelization.JobRunner` API provided by :mod:`pytools`
- BUILD: add support for :mod:`shap` 0.38 and 0.39


FACET 1.0
---------

1.0.3
~~~~~

- FIX: restrict package requirements to *gamma-pytools* 1.0.* and *sklearndf* 1.0.x, since FACET 1.0 is not compatible with *gamma-pytools* 1.1.* 

1.0.2
~~~~~

This is a maintenance release focusing on enhancements to the CI/CD pipeline and bug
fixes.

- API: add support for :mod:`shap` 0.36 and 0.37 via a new :class:`.BaseExplainer`
  stub class
- FIX: apply color scheme to the histogram section in :class:`.SimulationMatplotStyle`
- BUILD: add support for :mod:`numpy` 1.20
- BUILD: updates and changes to the CI/CD pipeline


1.0.1
~~~~~

Initial release.
