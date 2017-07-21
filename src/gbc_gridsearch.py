from stabilitasfinder import StabilitasFinder
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
from datetime import datetime
from sklearn.metrics import auc, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import json
plt.style.use("ggplot")


class VectorsToDense(object):
    """
    Class to help pass outputs from TF-IDF Vectorizer to Random Forest
    Classifier. Converts sparse matrix to dense in a way that fits with
    SKLearn pipeline.
    """
    def __init__(self):
        pass

    def fit(self, X, y, **kwargs):
        pass

    def fit_transform(self, X, y, **kwargs):
        return X.toarray()

    def transform(self, X, **kwargs):
        return X.toarray()

    def get_params(self, **kwargs):
        return {}


def main():
    """
    Main function to run grid search for hyperparameters. Prior results are
    saved as comments.
    """
    finder = StabilitasFinder()
    finder.load_data(source="data/flagged_reports.csv",)
    finder.label_critical_reports(cutoff=30)
    X = finder.flagged_df["title"].values
    y = finder.flagged_df["critical"].values

    # vectorizer = TfidfVectorizer(
    #     analyzer="word",
    #     stop_words="english",
    #     max_features=2500
    # )
    # X = vectorizer.fit_transform(X)
    # X = X.toarray()

    pipe = Pipeline([
        ("vectorizer", TfidfVectorizer(
                analyzer="word",
                stop_words="english",
                max_features=2500
            )),
        ("condenser", VectorsToDense()),
        ("classifier", GradientBoostingClassifier())
    ])

    # param_grid1 = {
    #     "classifier__learning_rate": [0.1],
    #     "classifier__n_estimators": [100],
    #     "classifier__max_depth": [1, 5, 9, 13],
    #     "classifier__min_samples_split": [2,1000, 2000],
    #     "classifier__min_samples_leaf": [1, 35, 70],
    #     "classifier__max_features": ["sqrt"],
    #     "classifier__subsample": [0.8]
    # }
    # grid1 = GridSearchCV(
    #     estimator=pipe,
    #     cv=5,
    #     n_jobs=-1,
    #     param_grid=param_grid1,
    #     scoring = "roc_auc",
    #     refit=False
    # )
    #
    # grid1.fit(X, y)
    # print "Best AUC: ", grid1.best_score_
    # print "Best params: ", grid1.best_params_

    # mean_scores = np.array(grid1.cv_results_["mean_test_score"])
    # fig, ax =plt.subplots(1, figsize=(8,8))
    # ax.plot(mean_scores, label="AUC scores")
    # plt.show()

    # Grid 1 Results:
    # Best AUC:  0.788662939265
    # Best params:  {
    #     'classifier__learning_rate': 0.1,
    #     'classifier__n_estimators': 100,
    #     'classifier__max_depth': 13
    #     'classifier__min_samples_split': 2,
    #     'classifier__min_samples_leaf': 1,
    #     'classifier__max_features': 'sqrt',
    #     'classifier__subsample': 0.8,
    # }

    # param_grid2 = {
    #     "classifier__learning_rate": [0.1],
    #     "classifier__n_estimators": [100],
    #     "classifier__max_depth": [9, 13, 17, 100],
    #     "classifier__min_samples_split": [2, 10, 100],
    #     "classifier__min_samples_leaf": [1, 10, 100],
    #     "classifier__max_features": ["sqrt"],
    #     "classifier__subsample": [0.8, 1]
    # }
    # grid2 = GridSearchCV(
    #     estimator=pipe,
    #     cv=5,
    #     n_jobs=-1,
    #     param_grid=param_grid2,
    #     scoring = "roc_auc",
    #     refit=False
    # )
    #
    # grid2.fit(X, y)
    # print "Best AUC: ", grid2.best_score_
    # print "Best params: ", grid2.best_params_

    # Grid 2 Results:
    # Best AUC:  0.795183794908
    # Best params:  {
    #     'classifier__learning_rate': 0.1,
    #     'classifier__n_estimators': 100,
    #     'classifier__max_depth': 13
    #     'classifier__min_samples_split': 2,
    #     'classifier__min_samples_leaf': 1,
    #     'classifier__max_features': 'sqrt',
    #     'classifier__subsample': 0.8,
    # }

    # param_grid3 = {
    #     "classifier__learning_rate": [0.1],
    #     "classifier__n_estimators": [100],
    #     "classifier__max_depth": [13],
    #     "classifier__min_samples_split": [2, 4, 8],
    #     "classifier__min_samples_leaf": [1, 4, 8],
    #     "classifier__max_features": ["sqrt"],
    #     "classifier__subsample": [0.2, 0.5, 0.8]
    # }
    # grid3 = GridSearchCV(
    #     estimator=pipe,
    #     cv=5,
    #     n_jobs=-1,
    #     param_grid=param_grid3,
    #     scoring = "roc_auc",
    #     refit=False
    # )
    #
    # grid3.fit(X, y)
    # print "Best AUC: ", grid3.best_score_
    # print "Best params: ", grid3.best_params_

    # Grid 3 Results:
    # Best AUC:  0.795295139988
    # Best params:  {
    #     'classifier__learning_rate': 0.1,
    #     'classifier__n_estimators': 100,
    #     'classifier__max_depth': 13
    #     'classifier__min_samples_split': 8,
    #     'classifier__min_samples_leaf': 1,
    #     'classifier__max_features': 'sqrt',
    #     'classifier__subsample': 0.5,
    # }

    # Grid 3 Results (2nd run):
    # Best AUC:  0.790485659763
    # Best params:  {
    #     'classifier__learning_rate': 0.1,
    #     'classifier__n_estimators': 100,
    #     'classifier__max_depth': 13
    #     'classifier__min_samples_split': 4,
    #     'classifier__min_samples_leaf': 1,
    #     'classifier__max_features': 'sqrt',
    #     'classifier__subsample': 0.8,
    # }

    # param_grid4 = {
    #     "classifier__learning_rate": [0.1],
    #     "classifier__n_estimators": [100],
    #     "classifier__max_depth": [13],
    #     "classifier__min_samples_split": [5, 6, 7, 8, 9],
    #     "classifier__min_samples_leaf": [1],
    #     "classifier__max_features": ["sqrt"],
    #     "classifier__subsample": [0.4, 0.5, 0.6]
    # }
    # grid4 = GridSearchCV(
    #     estimator=pipe,
    #     cv=5,
    #     n_jobs=-1,
    #     param_grid=param_grid4,
    #     scoring = "roc_auc",
    #     refit=False
    # )
    #
    # grid4.fit(X, y)
    # print "Best AUC: ", grid4.best_score_
    # print "Best params: ", grid4.best_params_

    # Grid 4 Results:
    # Best AUC:  0.796263914279
    # Best params:  {
    #     'classifier__learning_rate': 0.1,
    #     'classifier__n_estimators': 100,
    #     'classifier__max_depth': 13
    #     'classifier__min_samples_split': 6,
    #     'classifier__min_samples_leaf': 1,
    #     'classifier__max_features': 'sqrt',
    #     'classifier__subsample': 0.4,
    # }

    # param_grid5 = {
    #     "classifier__learning_rate": [0.1],
    #     "classifier__n_estimators": [100],
    #     "classifier__max_depth": [13],
    #     "classifier__min_samples_split": [4, 6, 8],
    #     "classifier__min_samples_leaf": [1, 2, 3],
    #     "classifier__max_features": ["sqrt"],
    #     "classifier__subsample": [0.4, 0.5, 0.8]
    # }
    # grid5 = GridSearchCV(
    #     estimator=pipe,
    #     cv=5,
    #     n_jobs=-1,
    #     param_grid=param_grid5,
    #     scoring = "roc_auc",
    #     refit=False
    # )
    #
    # grid5.fit(X, y)
    # print "Best AUC: ", grid5.best_score_
    # print "Best params: ", grid5.best_params_

    # Grid 5 Results:
    # Best AUC:  0.793876064368
    # Best params:  {
    #     'classifier__learning_rate': 0.1,
    #     'classifier__n_estimators': 100,
    #     'classifier__max_depth': 13
    #     'classifier__min_samples_split': 6,
    #     'classifier__min_samples_leaf': 3,
    #     'classifier__max_features': 'sqrt',
    #     'classifier__subsample': 0.5,
    # }

    # param_grid6 = {
    #     "classifier__learning_rate": [0.1],
    #     "classifier__n_estimators": [100],
    #     "classifier__max_depth": [10, 11, 12, 13, 14, 15, 16],
    #     "classifier__min_samples_split": [2, 6],
    #     "classifier__min_samples_leaf": [1, 3],
    #     "classifier__max_features": ["sqrt"],
    #     "classifier__subsample": [0.5]
    # }
    # grid6 = GridSearchCV(
    #     estimator=pipe,
    #     cv=5,
    #     n_jobs=-1,
    #     param_grid=param_grid6,
    #     scoring = "roc_auc",
    #     refit=False
    # )
    #
    # grid6.fit(X, y)
    # print "Best AUC: ", grid6.best_score_
    # print "Best params: ", grid6.best_params_

    # Grid 6 Results:
    # Best AUC:  0.793876064368
    # Best params:  {
    #     'classifier__learning_rate': 0.1,
    #     'classifier__n_estimators': 100,
    #     'classifier__max_depth': 12
    #     'classifier__min_samples_split': 2,
    #     'classifier__min_samples_leaf': 1,
    #     'classifier__max_features': 'sqrt',
    #     'classifier__subsample': 0.5,
    # }

    # param_grid7 = {
    #     "classifier__learning_rate": [0.1],
    #     "classifier__n_estimators": [100],
    #     "classifier__max_depth": [11, 12, 13],
    #     "classifier__min_samples_split": [2, 3, 4],
    #     "classifier__min_samples_leaf": [1, 2, 3],
    #     "classifier__max_features": ["sqrt"],
    #     "classifier__subsample": [0.4, 0.5, 0.6]
    # }
    # grid7 = GridSearchCV(
    #     estimator=pipe,
    #     cv=5,
    #     n_jobs=-1,
    #     param_grid=param_grid7,
    #     scoring = "roc_auc",
    #     refit=False
    # )
    #
    # grid7.fit(X, y)
    # print "Best AUC: ", grid7.best_score_
    # print "Best params: ", grid7.best_params_

    # Grid 7 Results:
    # Best AUC:  0.797160829407
    # Best params:  {
    #     'classifier__learning_rate': 0.1,
    #     'classifier__n_estimators': 100,
    #     'classifier__max_depth': 11
    #     'classifier__min_samples_split': 2,
    #     'classifier__min_samples_leaf': 2,
    #     'classifier__max_features': 'sqrt',
    #     'classifier__subsample': 0.4,
    # }

    # param_grid8 = {
    #     "classifier__learning_rate": [0.1],
    #     "classifier__n_estimators": [100],
    #     "classifier__max_depth": [10, 11, 12],
    #     "classifier__min_samples_split": [2, 3, 4],
    #     "classifier__min_samples_leaf": [1, 2, 3],
    #     "classifier__max_features": ["sqrt"],
    #     "classifier__subsample": [0.3, 0.4, 0.5]
    # }
    # grid8 = GridSearchCV(
    #     estimator=pipe,
    #     cv=5,
    #     n_jobs=-1,
    #     param_grid=param_grid8,
    #     scoring = "roc_auc",
    #     refit=False
    # )
    #
    # grid8.fit(X, y)
    # print "Best AUC: ", grid8.best_score_
    # print "Best params: ", grid8.best_params_

    # Grid 8 Results:
    # Best AUC:  0.793718925575
    # Best params:  {
    #     'classifier__learning_rate': 0.1,
    #     'classifier__n_estimators': 100,
    #     'classifier__max_depth': 12
    #     'classifier__min_samples_split': 3,
    #     'classifier__min_samples_leaf': 1,
    #     'classifier__max_features': 'sqrt',
    #     'classifier__subsample': 0.3,
    # }

    # param_grid9 = {
    #     "classifier__learning_rate": [0.1],
    #     "classifier__n_estimators": [100],
    #     "classifier__max_depth": [11, 12, 13, 14],
    #     "classifier__min_samples_split": [2, 3, 4],
    #     "classifier__min_samples_leaf": [1, 2, 3],
    #     "classifier__max_features": ["sqrt"],
    #     "classifier__subsample": [0.2, 0.3, 0.4]
    # }
    # grid9 = GridSearchCV(
    #     estimator=pipe,
    #     cv=5,
    #     n_jobs=-1,
    #     param_grid=param_grid9,
    #     scoring = "roc_auc",
    #     refit=False
    # )
    #
    # grid9.fit(X, y)
    # print "Best AUC: ", grid9.best_score_
    # print "Best params: ", grid9.best_params_

    # Grid 9 Results:
    # Best AUC:  0.794140915907
    # Best params:  {
    #     'classifier__learning_rate': 0.1,
    #     'classifier__n_estimators': 100,
    #     'classifier__max_depth': 14
    #     'classifier__min_samples_split': 3,
    #     'classifier__min_samples_leaf': 2,
    #     'classifier__max_features': 'sqrt',
    #     'classifier__subsample': 0.3,
    # }

    param_grid10 = {
        "classifier__learning_rate": [0.1],
        "classifier__n_estimators": [100],
        "classifier__max_depth": [13, 14, 15, None],
        "classifier__min_samples_split": [2, 3, 4],
        "classifier__min_samples_leaf": [1, 2, 3],
        "classifier__max_features": ["sqrt"],
        "classifier__subsample": [0.3]
    }
    grid10 = GridSearchCV(
        estimator=pipe,
        cv=5,
        n_jobs=-1,
        param_grid=param_grid10,
        scoring="roc_auc",
        refit=False
    )

    grid10.fit(X, y)
    print "Best AUC: ", grid10.best_score_
    print "Best params: ", grid10.best_params_

    # Grid 10 Results:
    # Best AUC:  0.789909985552
    # Best params:  {
    #     'classifier__learning_rate': 0.1,
    #     'classifier__n_estimators': 100,
    #     'classifier__max_depth': 13
    #     'classifier__min_samples_split': 3,
    #     'classifier__min_samples_leaf': 1,
    #     'classifier__max_features': 'sqrt',
    #     'classifier__subsample': 0.3,
    # }


if __name__ == '__main__':
    main()
