import numpy as np
import pytest
from sklearn import svm, datasets, tree
from sklearn.model_selection import GridSearchCV
from yieldengine.modeling.validation import CircularCrossValidator
import warnings

# noinspection PyUnresolvedReferences
from tests.shared_fixtures import batch_table


def test_circular_cv_init(batch_table):
    # filter out warnings triggerd by sk-learn/numpy

    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


    # check erroneous inputs
    #   - test_ratio = 0
    with pytest.raises(expected_exception=ValueError):
        CircularCrossValidator(test_ratio=0.0)

    #   - test_ratio < 0
    with pytest.raises(expected_exception=ValueError):
        CircularCrossValidator(test_ratio=-0.0001)

    #   - test_ratio > 1
    with pytest.raises(expected_exception=ValueError):
        CircularCrossValidator(test_ratio=1.00001)


def test_get_train_test_splits_as_indices():

    test_folds = 200
    test_X = np.arange(0, 1000, 1)

    for use_bootstrapping in (False, True):

        my_cv = CircularCrossValidator(
            test_ratio=0.2, num_folds=test_folds, use_bootstrapping=use_bootstrapping
        )

        list_of_test_splits = list(
            my_cv._iter_test_indices(test_X))

        # assert we get right amount of folds
        assert len(list_of_test_splits) == test_folds

        # check correct ratio of test/train
        for test_set in list_of_test_splits:
            assert 0.19 < float(len(test_set) / len(test_X) < 0.21)

        list_of_test_splits_2 = list(
            my_cv._iter_test_indices(test_X)
        )

        assert len(list_of_test_splits) == len(
            list_of_test_splits_2
        ), "The number of folds should be stable!"

        for f1, f2 in zip(list_of_test_splits, list_of_test_splits_2):
            assert np.array_equal(f1, f2), "Fold indices should be stable!"


def test_circular_cv_with_sk_learn():
    # filter out warnings triggerd by sk-learn/numpy

    warnings.filterwarnings("ignore", message="numpy.dtype size changed")
    warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


    # load example data
    iris = datasets.load_iris()

    # define a yield-engine circular CV:
    my_cv = CircularCrossValidator(test_ratio=0.21, num_folds=50)

    # define parameters and model
    parameters = {"kernel": ("linear", "rbf"), "C": [1, 10]}
    svc = svm.SVC(gamma="scale")

    # use the defined my_cv circular CV within GridSearchCV:
    clf = GridSearchCV(svc, parameters, cv=my_cv)
    clf.fit(iris.data, iris.target)

    # test if the number of received folds is correct:
    assert clf.n_splits_ == 50, "50 folds should have been generated by the circular CV"

    assert clf.best_score_ > 0.85, "Expected a minimum score of 0.85"

    # define new paramters and a different model
    # use the defined my_cv circular CV again within GridSeachCV:
    parameters = {
        "criterion": ("gini", "entropy"),
        "max_features": ["sqrt", "auto", "log2"],
    }
    cl2 = GridSearchCV(tree.DecisionTreeClassifier(), parameters, cv=my_cv)
    cl2.fit(iris.data, iris.target)

    assert cl2.best_score_ > 0.85, "Expected a minimum score of 0.85"
