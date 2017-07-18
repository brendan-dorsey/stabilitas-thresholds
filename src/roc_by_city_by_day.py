from stabilitasfinder import StabilitasFinder
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
from datetime import datetime
from sklearn.metrics import auc, confusion_matrix, f1_score
import json
plt.style.use("ggplot")


def main():

    with open("data/outputs_2016/volume_scoring_1wk_window/filter_vol_date_lookup_1w.json", "r") as f:
        date_lookup = json.load(f)

    with open("data/outputs_2016/volume_scoring_1wk_window/filter_vol_city_lookup_1w.json", "r") as f:
        city_lookup = json.load(f)

    ########################################
    ########################################
    ##                                    ##
    ##  The code below is for generating  ##
    ##  a ROC curve and determining an    ##
    ##  optimal decision threshold        ##
    ##                                    ##
    ########################################
    ########################################

    fig, ax = plt.subplots(1, figsize=(8,8))
    # List of start dates to consider. Will run from that date to
    # the end of 2016.
    cutoffs = [
        # ("2016-12-01", "2017-01-01"),
        # ("2016-11-01", "2017-01-01"),
        # ("2016-10-21", "2017-01-01"),
        # ("2016-10-11", "2017-01-01"),
        # ("2016-10-01", "2017-01-01"),
        ("2016-01-01", "2017-01-01"),
    ]
    for start_date, end_date in cutoffs:
        finder = StabilitasFinder()
        finder.load_data(
            source="data/outputs_2016/volume_scoring_1wk_window/flagged_reports_vol_1w_full.csv",
            date_lookup=date_lookup,
            city_lookup=city_lookup
        )
        finder.trim_dates(start_date, end_date)
        finder.label_critical_reports()
        finder._labeled_critical_cities_by_day()



        # Various ranges of thresholds used in cross validation.
        thresholds = np.linspace(0, 1, 201)

        #### Thresholds to keep track of ####
        # Volume, [Nov:Dec] = 0.13
        # Quadratic, [Nov:Dec] = 0.14

        for threshold in thresholds:
            pass

        # models = ["nb", "gbc", "rfc"]
        # models = ["gbc", "rfc", "svm"]
        # models = ["gbc", "rfc", "logreg", "nb"]
        # models = ["rfc", "gbc"]
        models = ["rfc"]
        for model in models:
            false_positive_rates = []
            true_positive_rates = []
            f1_scores = []
            y_true = finder.flagged_df["critical"].values
            predictions = finder.cross_val_predict(
                            thresholds=thresholds,
                            model_type=model
                        )

            for i, predicted in enumerate(predictions):
                conf_mat = confusion_matrix(y_true, predicted)
                # Transpoition of sklearn confusion matrix to this format:
                # TP  FN
                # FP  TN
                conf_mat = [[conf_mat[1][1], conf_mat[1][0]], [conf_mat[0][1], conf_mat[0][0]]]
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
                    precision = float(conf_mat[0][0]) / (conf_mat[0][0] + conf_mat[1][0])
                except:
                    precision = 0
                if (precision + tpr) == 0:
                    f1 = 0
                else:
                    f1 = 2 * (precision * tpr) / (precision + tpr)
                f1_scores.append(f1)
                false_positive_rates.append(fpr)
                true_positive_rates.append(tpr)

                if (tpr > 0.7) & (fpr < 0.5):
                    print "Model: ", model
                    print "Start Date: ", start_date
                    print "Threshold: ", thresholds[i]
                    print "TPR/Recall: ", tpr
                    print "FPR: ", fpr
                    print "Precision: ", precision
                    print "F1 score: ", f1
                    print ""

            area = auc(false_positive_rates, true_positive_rates)

            ax.plot(
                false_positive_rates,
                true_positive_rates,
                label="Start: {0}    Area: {1:0.3f}".format(start_date, area)
            )

    #         ax.plot(
    #             thresholds,
    #             f1_scores,
    #             label="F1 Scores for {}".format(model),
    #             linestyle=":",
    #             alpha=0.5
    #             )
    #
    # ax.fill_between(
    #     np.arange(0.0, 0.3, 0.01),
    #     0.7,
    #     1,
    #     color="g",
    #     alpha=0.3
    # )
    #
    # ax.scatter(
    #     0.075,
    #     0.983,
    #     color="k",
    #     label="Current Per Day Performance"
    # )

    ax.plot([0,1], [0, 1], linestyle="--", color="k")
    ax.plot([0,1], [0.7, 0.7], linestyle=":", color="g", label="Target Recall per Report")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Data/Performance Challenge")
    plt.legend(loc="lower right")
    plt.show()



if __name__ == '__main__':
    main()
