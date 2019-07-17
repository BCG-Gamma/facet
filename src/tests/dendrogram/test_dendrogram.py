import pandas as pd
import pytest
from lightgbm import LGBMRegressor
from matplotlib.pyplot import figure

from yieldengine import Sample
from yieldengine.dendrogram import DendrogramDrawer
from yieldengine.dendrogram.style import FeatMapStyle, LineStyle
from yieldengine.df.transform import DataFrameTransformer
from yieldengine.model import RegressionModel
from yieldengine.model.inspection import ModelInspector
from yieldengine.model.prediction import ModelFitCV
from yieldengine.model.validation import CircularCrossValidator


@pytest.fixture()
def model_inspector(
    batch_table: pd.DataFrame, sample: Sample, simple_preprocessor: DataFrameTransformer
) -> ModelInspector:

    cv = CircularCrossValidator(test_ratio=0.20, num_splits=5)
    model = RegressionModel(
        estimator=LGBMRegressor(), preprocessing=simple_preprocessor
    )
    return ModelInspector(model_fit=ModelFitCV(model=model, cv=cv, sample=sample))


def test_linkage_drawer_style(model_inspector: ModelInspector) -> None:
    linkage = model_inspector.cluster_dependent_features()
    fig = figure(figsize=(8, 16))
    ax = fig.add_subplot(111)
    dd = DendrogramDrawer(title="Test", linkage_tree=linkage, style=LineStyle(ax=ax))
    dd.draw()
    dd_2 = DendrogramDrawer(
        title="Test", linkage_tree=linkage, style=FeatMapStyle(ax=ax)
    )
    dd_2.draw()
