"""
Model selection and hyperparameter optimization.

:class:`.LearnerGrid` encapsulates a :class:`.LearnerPipelineDF` and a grid of
hyperparameters.

:class:`.LearnerRanker` selects the best pipeline and parametrization based on the
pipeline and hyperparameter choices provided as a list of :class:`.LearnerGrid`.
"""
from ._selection import *
