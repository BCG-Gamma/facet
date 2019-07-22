from yieldengine import Sample
from yieldengine.sklearndf import DataFrameTransformer


def test_column_transformer_df(
    sample: Sample, simple_preprocessor: DataFrameTransformer
) -> None:
    simple_preprocessor.fit_transform(X=sample.features)
