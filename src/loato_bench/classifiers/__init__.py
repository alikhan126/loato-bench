"""Classifier implementations (LogReg, SVM, XGBoost, MLP)."""

from loato_bench.classifiers.base import Classifier
from loato_bench.classifiers.logreg import LogRegClassifier
from loato_bench.classifiers.mlp import MLPClassifier
from loato_bench.classifiers.svm import SVMClassifier
from loato_bench.classifiers.xgb import XGBoostClassifier

__all__ = [
    "Classifier",
    "LogRegClassifier",
    "MLPClassifier",
    "SVMClassifier",
    "XGBoostClassifier",
]
