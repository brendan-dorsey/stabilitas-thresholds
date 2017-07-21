from stabilitasfinder import StabilitasFinder
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
from datetime import datetime
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, auc
from itertools import combinations, product
import json
plt.style.use("ggplot")


def main():
    """
    Main function to run script.
    """
    roc_by_city_by_day(
        date_lookup="""data/outputs_2016/volume_scoring_1wk_window/
        filter_vol_date_lookup_1wk.json"""
        city_lookup="""data/outputs_2016/volume_scoring_1wk_window/
        filter_vol_city_lookup_1wk.json"""
        data_source="""data/outputs_2016/volume_scoring_1wk_window/
        filter_vol_flagged_reports_1wk.csv"""
    )


def roc_by_city_by_day(date_lookup, city_lookup, data_source):
    """
    Function to generate ROC curve on a per-event proxy basis, from the Filter
    layer.

    Inputs:
    date_lookup - str, filepath to date lookup output from filter layer.
    city_lookup - str, filepath to city lookup output from filter layer.
    data_source - str, filepath to flagged reports output from filter layer.
    """

    with open(date_lookup, "r") as f:
        date_lookup = json.load(f)

    with open(city_lookup, "r") as f:
        city_lookup = json.load(f)

    ########################################
    ########################################
    #                                      #
    #   The code below is for generating   #
    #   a ROC curve and determining an     #
    #   optimal decision threshold         #
    #                                      #
    ########################################
    ########################################

    fig, ax = plt.subplots(1, figsize=(8, 8))

    finder = StabilitasFinder()
    finder.load_data(
        source=data_source,
        date_lookup=date_lookup,
        city_lookup=city_lookup
    )

    finder.trim_dates("2016-01-01", "2017-01-01")
    finder.label_critical_reports()
    finder._labeled_critical_cities_by_day()

    false_positive_rates = []
    true_positive_rates = []
    f1_scores = []

    # thresholds = np.linspace(0, 1, 201)
    # thresholds = [0.14, 0.22]
    thresholds = list(np.arange(0.1, 0.25, step=0.01))
    thresholds.append(1)
    thresholds.insert(0, 0)

    for threshold in thresholds:
        print "     Checking threshold = ", threshold
        try:
            del finder.flagged_df["predicted"]
        except:
            pass
        finder.date_lookup = date_lookup
        finder.city_lookup = city_lookup

        finder.cross_val_predict(thresholds=[threshold], model_type="rfc")
        finder._labeled_critical_cities_by_day()
        finder._predicted_critical_cities_by_day()
        finder._most_critical_report_per_city_per_day()

        temp_date_lookup = finder.date_lookup
        dates = temp_date_lookup.keys()

        cities = set()
        for date in dates:
            try:
                for city in temp_date_lookup[date][0]:
                    cities.add(city)
            except IndexError:
                continue

        city_date_pairs = product(cities, dates)

        y_true = []
        y_pred = []
        for city, date in city_date_pairs:
            try:
                if city in temp_date_lookup[date][1]:
                    y_true.append(1)
                else:
                    y_true.append(0)
                if city in temp_date_lookup[date][2]:
                    y_pred.append(1)
                else:
                    y_pred.append(0)
            except:
                continue

        conf_mat = confusion_matrix(y_true, y_pred)
        # Transpoition of sklearn confusion matrix to this format:
        # TP  FN
        # FP  TN
        conf_mat = [
            [conf_mat[1][1], conf_mat[1][0]],
            [conf_mat[0][1], conf_mat[0][0]]
        ]

        # True Positive Rate: TP / TP + FN
        try:
            tpr = float(conf_mat[0][0]) / (conf_mat[0][0] + conf_mat[0][1])
        except:
            tpr = 0

        # False Positive Rate: FP / FP + TN
        try:
            fpr = float(conf_mat[1][0]) / (conf_mat[1][0] + conf_mat[1][1])
        except:
            fpr = 0

        # Precision: TP / TP + FP
        try:
            precision = (float(conf_mat[0][0]) /
                         (conf_mat[0][0] + conf_mat[1][0]))
        except:
            precision = 0

        # False Discovery Rate: FP / TP + FP
        try:
            fdr = float(conf_mat[1][0]) / (conf_mat[0][0] + conf_mat[1][0])
        except:
            fdr = 0

        if (precision + tpr) == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * tpr) / (precision + tpr)

        true_positive_rates.append(tpr)
        false_positive_rates.append(fpr)
        f1_scores.append(f1)

        # print "Threshold: ", threshold
        # print "TPR/Recall: ", tpr
        # print "FPR: ", fpr
        # print "Precision: ", precision
        # print "F1 score: ", f1
        # print ""

        if (tpr > 0.9):
            print "Threshold: ", threshold
            print "TPR/Recall: ", tpr
            print "FPR: ", fpr
            print "Precision: ", precision
            print "F1 score: ", f1
            print ""

    area = auc(false_positive_rates, true_positive_rates)

    ax.plot(
        false_positive_rates,
        true_positive_rates,
        label="Per City Per Day    Area: {0:0.3f}".format(area)
    )

    ax.plot(
        thresholds,
        f1_scores,
        label="F1 Scores",
        linestyle=":",
        )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="k",
    )
    ax.plot(
        [0, 1],
        [0.95, 0.95],
        linestyle=":",
        color="g",
        label="Target Recall per City per Day",
    )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Per City Per Day ROC Curve")
    plt.legend(loc="lower right")
    plt.show()


if __name__ == '__main__':
    main()
