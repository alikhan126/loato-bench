"""Tests for all classifier implementations."""

import numpy as np
import pytest

from loato_bench.classifiers.base import Classifier
from loato_bench.classifiers.logreg import LogRegClassifier
from loato_bench.classifiers.mlp import MLPClassifier
from loato_bench.classifiers.svm import SVMClassifier
from loato_bench.classifiers.xgb import XGBoostClassifier

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def toy_data():
    """Linearly separable toy dataset for quick classifier tests."""
    rng = np.random.RandomState(42)
    X_pos = rng.randn(50, 8).astype(np.float32) + 2.0
    X_neg = rng.randn(50, 8).astype(np.float32) - 2.0
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * 50 + [0] * 50, dtype=np.int64)
    return X, y


ALL_CLASSIFIERS: list[type[Classifier]] = [
    LogRegClassifier,
    SVMClassifier,
    XGBoostClassifier,
    MLPClassifier,
]


# ---------------------------------------------------------------------------
# Contract tests — ABC compliance
# ---------------------------------------------------------------------------


class TestClassifierContract:
    """Verify all classifiers satisfy the Classifier ABC."""

    @pytest.mark.parametrize("cls", ALL_CLASSIFIERS)
    def test_is_subclass(self, cls: type[Classifier]):
        assert issubclass(cls, Classifier)

    @pytest.mark.parametrize("cls", ALL_CLASSIFIERS)
    def test_has_name_property(self, cls: type[Classifier]):
        instance = cls()
        assert isinstance(instance.name, str)
        assert len(instance.name) > 0

    @pytest.mark.parametrize("cls", ALL_CLASSIFIERS)
    def test_has_fit_method(self, cls: type[Classifier]):
        assert callable(getattr(cls, "fit", None))

    @pytest.mark.parametrize("cls", ALL_CLASSIFIERS)
    def test_has_predict_method(self, cls: type[Classifier]):
        assert callable(getattr(cls, "predict", None))

    @pytest.mark.parametrize("cls", ALL_CLASSIFIERS)
    def test_has_predict_proba_method(self, cls: type[Classifier]):
        assert callable(getattr(cls, "predict_proba", None))


# ---------------------------------------------------------------------------
# Naming tests
# ---------------------------------------------------------------------------


class TestClassifierNames:
    """Verify each classifier reports the expected name."""

    def test_logreg_name(self):
        assert LogRegClassifier().name == "logreg"

    def test_svm_name(self):
        assert SVMClassifier().name == "svm"

    def test_xgboost_name(self):
        assert XGBoostClassifier().name == "xgboost"

    def test_mlp_name(self):
        assert MLPClassifier().name == "mlp"


# ---------------------------------------------------------------------------
# Functional tests — fit / predict / predict_proba
# ---------------------------------------------------------------------------


class TestClassifierFit:
    """Test that classifiers can fit on toy data without errors."""

    @pytest.mark.parametrize("cls", ALL_CLASSIFIERS)
    def test_fit_runs(self, cls: type[Classifier], toy_data):
        X, y = toy_data
        clf = cls()
        clf.fit(X, y)  # Should not raise


class TestClassifierPredict:
    """Test predict output shape and dtype."""

    @pytest.mark.parametrize("cls", ALL_CLASSIFIERS)
    def test_predict_shape(self, cls: type[Classifier], toy_data):
        X, y = toy_data
        clf = cls()
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (len(X),)

    @pytest.mark.parametrize("cls", ALL_CLASSIFIERS)
    def test_predict_dtype(self, cls: type[Classifier], toy_data):
        X, y = toy_data
        clf = cls()
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.dtype == np.int64

    @pytest.mark.parametrize("cls", ALL_CLASSIFIERS)
    def test_predict_values_binary(self, cls: type[Classifier], toy_data):
        X, y = toy_data
        clf = cls()
        clf.fit(X, y)
        preds = clf.predict(X)
        assert set(np.unique(preds)).issubset({0, 1})


class TestClassifierPredictProba:
    """Test predict_proba output shape, dtype, and probability constraints."""

    @pytest.mark.parametrize("cls", ALL_CLASSIFIERS)
    def test_proba_shape(self, cls: type[Classifier], toy_data):
        X, y = toy_data
        clf = cls()
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (len(X), 2)

    @pytest.mark.parametrize("cls", ALL_CLASSIFIERS)
    def test_proba_dtype(self, cls: type[Classifier], toy_data):
        X, y = toy_data
        clf = cls()
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.dtype == np.float64

    @pytest.mark.parametrize("cls", ALL_CLASSIFIERS)
    def test_proba_sums_to_one(self, cls: type[Classifier], toy_data):
        X, y = toy_data
        clf = cls()
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    @pytest.mark.parametrize("cls", ALL_CLASSIFIERS)
    def test_proba_between_zero_and_one(self, cls: type[Classifier], toy_data):
        X, y = toy_data
        clf = cls()
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)


# ---------------------------------------------------------------------------
# Accuracy tests — should get near-perfect on linearly separable data
# ---------------------------------------------------------------------------


class TestClassifierAccuracy:
    """Verify classifiers achieve high accuracy on easy toy data."""

    @pytest.mark.parametrize("cls", ALL_CLASSIFIERS)
    def test_high_accuracy_on_separable_data(self, cls: type[Classifier], toy_data):
        X, y = toy_data
        clf = cls()
        clf.fit(X, y)
        preds = clf.predict(X)
        accuracy = (preds == y).mean()
        assert accuracy >= 0.95, f"{cls.__name__} accuracy {accuracy:.2f} < 0.95"


# ---------------------------------------------------------------------------
# Custom hyperparameter tests
# ---------------------------------------------------------------------------


class TestClassifierHyperparams:
    """Test that classifiers accept custom hyperparameters."""

    def test_logreg_custom_C(self, toy_data):
        X, y = toy_data
        clf = LogRegClassifier(C=0.01, max_iter=500, solver="lbfgs")
        clf.fit(X, y)
        assert clf.predict(X).shape == (len(X),)

    def test_svm_custom_params(self, toy_data):
        X, y = toy_data
        clf = SVMClassifier(C=10.0, kernel="rbf", gamma="auto")
        clf.fit(X, y)
        assert clf.predict(X).shape == (len(X),)

    def test_xgboost_custom_params(self, toy_data):
        X, y = toy_data
        clf = XGBoostClassifier(n_estimators=50, max_depth=3, learning_rate=0.1)
        clf.fit(X, y)
        assert clf.predict(X).shape == (len(X),)

    def test_mlp_custom_params(self, toy_data):
        X, y = toy_data
        clf = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            learning_rate_init=0.01,
            max_iter=100,
        )
        clf.fit(X, y)
        assert clf.predict(X).shape == (len(X),)
