#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/25 8:03
# @Author  : huag@kth.se
# @File    : gbdt_tree.py
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import classification_report


class GradientBoost:

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass


if __name__ == "__main__":
    ###### Config##########
    n_estimators = 5
    learning_rate = 1
    #######################

    data = datasets.load_iris()
    X = data.data
    Y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.8)
    model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Scikit-learn  AdaBoostClassifier:")
    # print(f"    Base estimator errors: {model.estimator_errors_}")
    # print(f"    Base estimator weights: {model.estimator_weights_}")
    print("     Classification report is :")
    print(classification_report(y_true=y_test, y_pred=y_pred))
    # model = GradientBoost(n_estimators=n_estimators, learning_rate=learning_rate)
    # model.fit(X_train, y_train)
    # # print("Machine learning glossary Adaboost classifier:")
    # # print(f"    Base estimator errors: {model.errors}")
    # # print(f"    Base estimatro weights: {model.estimator_weights}")
    # y_pred = model.predict(X_test)
    # print("     Classification report is :")
    # print(classification_report(y_true=y_test, y_pred=y_pred))