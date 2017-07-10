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

    pipe = Pipeline([
        ("vectorizer", TfidfVectorizer()),
        ("classifier", GradientBoostingClassifier())
    ])

    param_grid1 = {
        "vectorizer__analyzer": ["word"],
        "vectorizer__stop_words": ["english"],
        "vectorizer__dtype": ["array"],
        "vectorizer__max_features": [300, 500, 800, 2500],
        "classifier__learning_rate": [0.1],
        "classifier__n_estimators": [100],
        "classifier__max_depth": [1, 5, 9, 13],
        "classifier__min_samples_split": [2, 100, 1000, 2000],
        "classifier__min_samples_leaf": [1, 35, 70],
        "classifier__max_features": ["sqrt", "log2", None],
        "classifier__subsample": [0.8, 1]
    }
    grid1 = GridSearchCV(
        estimator=pipe,
        cv=5,
        n_jobs=-1,
        param_grid=param_grid1,
        scoring = "roc_auc",
        refit=False
    )

    grid1.fit(X, y)
    print "Best AUC: ", grid1.best_score_
    print "Best params: ", grid1.best_params_
    mean_scores = np.array(grid1.cv_results_["mean_test_score"])
    fig, ax =plt.subplots(1, figsize=(8,8))
    ax.plot(mean_scores, label="AUC scores")
    plt.show()





if __name__ == '__main__':
    main()
