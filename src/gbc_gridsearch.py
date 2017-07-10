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


def main():
    finder = StabilitasFinder()
    finder.load_data(source="data/flagged_reports.csv",)
    finder.label_critical_reports(cutoff=30)
    X = finder.flagged_df["title"].values
    y = finder.flagged_df["critical"].values

    vectorizer = TfidfVectorizer(
        analyzer="word",
        stop_words="english",
        max_features=2500
    )
    X = vectorizer.fit_transform(X)
    X = X.toarray()

    pipe = Pipeline([
        ("classifier", GradientBoostingClassifier())
    ])

    # Takes WAAAY to long to run.
    # param_grid1 = {
    #     "vectorizer__analyzer": ["word"],
    #     "vectorizer__stop_words": ["english"],
    #     "vectorizer__dtype": ["array"],
    #     "vectorizer__max_features": [300, 500, 800, 2500],
    #     "classifier__learning_rate": [0.1],
    #     "classifier__n_estimators": [100],
    #     "classifier__max_depth": [1, 5, 9, 13],
    #     "classifier__min_samples_split": [2, 100, 1000, 2000],
    #     "classifier__min_samples_leaf": [1, 35, 70],
    #     "classifier__max_features": ["sqrt", "log2", None],
    #     "classifier__subsample": [0.8, 1]
    # }

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

    param_grid3 = {
        "classifier__learning_rate": [0.1],
        "classifier__n_estimators": [100],
        "classifier__max_depth": [13],
        "classifier__min_samples_split": [2, 4, 8],
        "classifier__min_samples_leaf": [1, 4, 8],
        "classifier__max_features": ["sqrt"],
        "classifier__subsample": [0.2, 0.5, 0.8]
    }
    grid3 = GridSearchCV(
        estimator=pipe,
        cv=5,
        n_jobs=-1,
        param_grid=param_grid3,
        scoring = "roc_auc",
        refit=False
    )

    grid3.fit(X, y)
    print "Best AUC: ", grid3.best_score_
    print "Best params: ", grid3.best_params_

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



if __name__ == '__main__':
    main()
