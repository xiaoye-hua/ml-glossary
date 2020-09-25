from abc import ABCMeta
from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
import numpy as np


class BoostedClassifier(metaclass=ABCMeta):
    """
    Base class for both weight-boosted tree and gradient-boosted tree
    """
    def __init__(self, n_estimators, learning_rate, base_estimator=DecisionTreeClassifier):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.estimators = [1 for _ in range(self.n_estimators)]
        self.errors = [1 for _ in range(self.n_estimators)]
        self.estimator_weights = [1 for _ in range(self.n_estimators)]
        self.base_estimator = base_estimator
        self.n_classes = None

    @classmethod
    def fit(self, X, y):
        pass

    @classmethod
    def predict(self, X):
        pass

    @classmethod
    def _boost(self, estimator_idx, X, y, sample_weight):
        pass


class AdaBoost(BoostedClassifier):

    def fit(self, X, y):
        self.n_classes = len(set(y))
        for estimator_idx in range(self.n_estimators):
            if estimator_idx == 0:
                sample_weight = self._init_sample_weight(X)
            sample_weight, estimator_weight, estimator_error = self._boost(
                estimator_idx=estimator_idx,
                X = X,
                y=y,
                sample_weight=sample_weight
            )

    def predict(self, X):
        weight_prediction = []
        for estimator, weight in zip(self.estimators, self.estimator_weights):
            raw_prediction = estimator.predict(X)
            prediction = []
            for p in raw_prediction:
                one_hot = [0 for _ in range(self.n_classes)]
                one_hot[p] = 1
                prediction.append(one_hot)
            prediction = np.array(prediction)
            weight_prediction.append(prediction * weight)
        weight_prediction = sum(weight_prediction)/sum(self.estimator_weights)
        result = np.argmax(weight_prediction, axis=1)
        return result

    def _init_sample_weight(self, X):
        sample_num = X.shape[0]
        return np.array([1/float(sample_num) for _ in range(sample_num)])

    def _boost(self, estimator_idx, X, y, sample_weight):
        estimator = self.base_estimator(max_depth=1)
        estimator.fit(X, y, sample_weight=sample_weight)
        y_pred = estimator.predict(X)
        incorrect = y != y_pred
        estimator_error = np.average(
            incorrect, weights=sample_weight, axis=0
        )
        estimator_weight = self.learning_rate * (
                np.log((1. - estimator_error) / estimator_error) +
                np.log(self.n_classes - 1.))
        sample_weight *= np.exp(estimator_weight * incorrect *
                                (sample_weight > 0))
        self.estimators[estimator_idx] = estimator
        self.estimator_weights[estimator_idx] = estimator_weight
        self.errors[estimator_idx] = estimator_error
        return sample_weight, estimator_weight, estimator_error


if __name__ == "__main__":
    ###### Config##########
    n_estimators = 5
    learning_rate = 1
    #######################

    data = datasets.load_iris()
    X = data.data
    Y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.8)
    model = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, algorithm="SAMME")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Scikit-learn  AdaBoostClassifier:")
    print(f"    Base estimator errors: {model.estimator_errors_}")
    print(f"    Base estimator weights: {model.estimator_weights_}")
    print("     Classification report is :")
    print(classification_report(y_true=y_test, y_pred=y_pred))
    model = AdaBoost(n_estimators=n_estimators, learning_rate=learning_rate)
    model.fit(X_train, y_train)
    print("Machine learning glossary Adaboost classifier:")
    print(f"    Base estimator errors: {model.errors}")
    print(f"    Base estimatro weights: {model.estimator_weights}")
    y_pred = model.predict(X_test)
    print("     Classification report is :")
    print(classification_report(y_true=y_test, y_pred=y_pred))
