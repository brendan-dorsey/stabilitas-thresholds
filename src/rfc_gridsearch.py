from stabilitasfinder import StabilitasFinder
import pandas as pd
import numpy as np
import time
from datetime import datetime
from sklearn.metrics import auc, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import json


def main():
    finder = StabilitasFinder()
    finder.load_data(source="debug/DEC_subset/flagged_reports_quad_1wk.csv")
    finder.label_critical_reports(cutoff=30)
    X = finder.flagged_df["title"].values
    y = finder.flagged_df["critical"].values

    pipe = Pipeline([
        ("vectorizer", TfidfVectorizer()),
        ("classifier", RandomForestClassifier())
    ])

    # param_grid1 = {
    #     "vectorizer__analyzer": ["word"],
    #     "vectorizer__stop_words": ["english"],
    #     "vectorizer__max_features": [300, 500, 800, 2500],
    #     "classifier__n_estimators": [100],
    #     "classifier__max_depth": [1, 5, 9, 13, None],
    #     "classifier__min_samples_split": [2, 100, 1000, 2000],
    #     "classifier__min_samples_leaf": [1, 35, 70],
    #     "classifier__max_features": ["sqrt", "log2", None],
    #     "classifier__n_jobs": [-1]
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
    #
    # print "Best AUC: ", grid1.best_score_
    # print "Best params: ", grid1.best_params_

    # mean_scores = np.array(grid1.cv_results_["mean_test_score"])
    # fig, ax =plt.subplots(1, figsize=(8,8))
    # ax.plot(mean_scores, label="AUC scores")
    # plt.show()

    # Grid 1 Results:
    # Best AUC:  0.792678915537
    # Best params:  {
    #     'vectorizer__stop_words': 'english',
    #     'vectorizer__analyzer': 'word',
    #     'vectorizer__max_features': 2500,
    #     'classifier__n_estimators': 100,
    #     'classifier__max_depth': None
    #     'classifier__min_samples_split': 2000,
    #     'classifier__min_samples_leaf': 1,
    #     'classifier__max_features': 'sqrt',
    #     'classifier__n_jobs': -1,
    # }

    # param_grid2 = {
    #     "vectorizer__analyzer": ["word"],
    #     "vectorizer__stop_words": ["english"],
    #     "vectorizer__max_features": [2500],
    #     "classifier__n_estimators": [100],
    #     "classifier__max_depth": [None],
    #     "classifier__min_samples_split": [2000, 2100, 2200, 2300, 2400, 2500],
    #     "classifier__min_samples_leaf": [1, 2, 3, 4, 5],
    #     "classifier__max_features": ["sqrt"],
    #     "classifier__n_jobs": [-1]
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
    #
    # print "Best AUC: ", grid2.best_score_
    # print "Best params: ", grid2.best_params_

    # Grid 2 Results:
    # Best AUC:  0.791779190603
    # Best params:  {
    #     'vectorizer__analyzer': 'word',
    #     'vectorizer__stop_words': 'english',
    #     'vectorizer__max_features': 2500,
    #     'classifier__n_estimators': 100,
    #     'classifier__max_depth': None
    #     'classifier__min_samples_split': 2200,
    #     'classifier__min_samples_leaf': 2,
    #     'classifier__max_features': 'sqrt',
    #     'classifier__n_jobs': -1,
    # }

    # param_grid3 = {
    #     "vectorizer__analyzer": ["word"],
    #     "vectorizer__stop_words": ["english"],
    #     "vectorizer__max_features": [2500],
    #     "classifier__n_estimators": [100],
    #     "classifier__max_depth": [None],
    #     "classifier__min_samples_split": [2, 2200],
    #     "classifier__min_samples_leaf": [1, 2, 3, 4, 5],
    #     "classifier__max_features": ["sqrt"],
    #     "classifier__n_jobs": [-1]
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
    #
    # print "Best AUC: ", grid3.best_score_
    # print "Best params: ", grid3.best_params_

    # Grid 3 Results:
    # Best AUC:  0.792287096279
    # Best params:  {
    #     'vectorizer__analyzer': 'word',
    #     'vectorizer__stop_words': 'english',
    #     'vectorizer__max_features': 2500,
    #     'classifier__n_estimators': 100,
    #     'classifier__max_depth': None
    #     'classifier__min_samples_split': 2,
    #     'classifier__min_samples_leaf': 3,
    #     'classifier__max_features': 'sqrt',
    #     'classifier__n_jobs': -1,
    # }

    # param_grid4 = {
    #     "vectorizer__analyzer": ["word"],
    #     "vectorizer__stop_words": ["english"],
    #     "vectorizer__max_features": [2500],
    #     "classifier__n_estimators": [100],
    #     "classifier__max_depth": [None],
    #     "classifier__min_samples_split": [2, 3, 4, 5],
    #     "classifier__min_samples_leaf": [1, 2, 3, 4, 5],
    #     "classifier__max_features": ["sqrt"],
    #     "classifier__n_jobs": [-1]
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
    #
    # print "Best AUC: ", grid4.best_score_
    # print "Best params: ", grid4.best_params_

    # Grid 4 Results:
    # Best AUC:  0.794431236878
    # Best params:  {
    #     'vectorizer__analyzer': 'word',
    #     'vectorizer__stop_words': 'english',
    #     'vectorizer__max_features': 2500,
    #     'classifier__n_estimators': 100,
    #     'classifier__max_depth': None
    #     'classifier__min_samples_split': 5,
    #     'classifier__min_samples_leaf': 3,
    #     'classifier__max_features': 'sqrt',
    #     'classifier__n_jobs': -1,
    # }

    # param_grid5 = {
    #     "vectorizer__analyzer": ["word"],
    #     "vectorizer__stop_words": ["english"],
    #     "vectorizer__max_features": [2500],
    #     "classifier__n_estimators": [100],
    #     "classifier__max_depth": [1, 10, 100, None],
    #     "classifier__min_samples_split": [2, 5, 10, 100, 1000],
    #     "classifier__min_samples_leaf": [3],
    #     "classifier__max_features": ["sqrt"],
    #     "classifier__n_jobs": [-1]
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
    #
    # print "Best AUC: ", grid5.best_score_
    # print "Best params: ", grid5.best_params_

    # Grid 5 Results:
    # Best AUC:  0.794240700912
    # Best params:  {
    #     'vectorizer__analyzer': 'word',
    #     'vectorizer__stop_words': 'english',
    #     'vectorizer__max_features': 2500,
    #     'classifier__n_estimators': 100,
    #     'classifier__max_depth': None
    #     'classifier__min_samples_split': 2,
    #     'classifier__min_samples_leaf': 3,
    #     'classifier__max_features': 'sqrt',
    #     'classifier__n_jobs': -1,
    # }

    # param_grid6 = {
    #     "vectorizer__analyzer": ["word"],
    #     "vectorizer__stop_words": ["english"],
    #     "vectorizer__max_features": [2500],
    #     "classifier__n_estimators": [200],
    #     "classifier__max_depth": [None],
    #     "classifier__min_samples_split": [2, 5, 10, 100],
    #     "classifier__min_samples_leaf": [1, 2, 3, 10],
    #     "classifier__max_features": ["sqrt", "log2", 25, 100],
    #     "classifier__n_jobs": [-1]
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
    #
    # print "Best AUC: ", grid6.best_score_
    # print "Best params: ", grid6.best_params_

    # with open("data/best_params.json", mode="w") as f:
    #     json.dump(grid6.best_params_, f)
    #
    # with open("data/best_score.json", mode="w") as f:
    #     json.dump(grid6.best_score_, f)

    # Grid 6 Results:
    # Best AUC:  0.561452266293
    # Best params:  {
    #     'vectorizer__analyzer': 'word',
    #     'vectorizer__stop_words': 'english',
    #     'vectorizer__max_features': 2500,
    #     'classifier__n_estimators': 100,
    #     'classifier__max_depth': None
    #     'classifier__min_samples_split': 10,
    #     'classifier__min_samples_leaf': 1,
    #     'classifier__max_features': '100',
    #     'classifier__n_jobs': -1,
    # }


    param_grid7 = {
        "vectorizer__analyzer": ["word"],
        "vectorizer__stop_words": ["english"],
        "vectorizer__max_features": [2500],
        "classifier__n_estimators": [200],
        "classifier__max_depth": [None],
        "classifier__min_samples_split": [2, 5, 10, 100],
        "classifier__min_samples_leaf": [1],
        "classifier__max_features": [100],
        "classifier__n_jobs": [-1]
    }
    grid7 = GridSearchCV(
        estimator=pipe,
        cv=5,
        n_jobs=-1,
        param_grid=param_grid7,
        scoring = "f1",
        refit=False
    )

    grid7.fit(X, y)

    print "Best AUC: ", grid7.best_score_
    print "Best params: ", grid7.best_params_

    # with open("data/best_params.json", mode="w") as f:
    #     json.dump(grid6.best_params_, f)
    #
    # with open("data/best_score.json", mode="w") as f:
    #     json.dump(grid6.best_score_, f)

    # Grid 7 Results:
    # Best AUC:  0.561452266293
    # Best params:  {
    #     'vectorizer__analyzer': 'word',
    #     'vectorizer__stop_words': 'english',
    #     'vectorizer__max_features': 2500,
    #     'classifier__n_estimators': 100,
    #     'classifier__max_depth': None
    #     'classifier__min_samples_split': 10,
    #     'classifier__min_samples_leaf': 1,
    #     'classifier__max_features': '100',
    #     'classifier__n_jobs': -1,
    # }


if __name__ == '__main__':
    main()
