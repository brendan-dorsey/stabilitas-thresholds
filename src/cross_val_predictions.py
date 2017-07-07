from stabilitasfinder import StabilitasFinder
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
from datetime import datetime
from sklearn.metrics import auc, confusion_matrix
import json
plt.style.use("ggplot")


def main():

    with open("app/date_lookup.json", "r") as f:
        date_lookup = json.load(f)

    with open("app/city_lookup.json", "r") as f:
        city_lookup = json.load(f)


    fig, ax = plt.subplots(1, figsize=(8,8))
    cutoffs = [10, 20, 30]
    for i, cutoff in enumerate(cutoffs):
        finder = StabilitasFinder()
        finder.load_data(
        source="data/flagged_reports.csv",
        date_lookup=date_lookup,
        city_lookup=city_lookup
        )
        finder.label_critical_reports(cutoff)
        finder._labeled_critical_cities_by_day()

        # finder.cross_val_predict()
        # finder._predicted_critical_cities_by_day()
        # with open("app/cv_date_lookup.json", "w") as f:
        #     json.dump(finder.date_lookup, f)

        ########################################
        ########################################
        ##                                    ##
        ##  The code below is for generating  ##
        ##  a ROC curve and determining an    ##
        ##  optimal decision threshold (0.235)##
        ##                                    ##
        ########################################
        ########################################


        thresholds = np.linspace(0, 1, 99)
        # thresholds = [0.22, 0.225, 0.23, 0.235, 0.24]
        # thresholds = [0.235]
        # thresholds = [0.45, 0.47, 0.49, 0.51, 0.53, 0.55]

        models = ["nb", "gbc", "rfc"]
        for model in models:
            false_positive_rates = []
            true_positive_rates = []
            y_true = finder.flagged_df["critical"].values

            for i, predicted in enumerate(finder.cross_val_predict(thresholds=thresholds, model_type=model)):
                conf_mat = confusion_matrix(y_true, predicted)
                # Transpoition of sklearn confusion matrix to my preferred format:
                # TP  FN
                # FP  TN
                conf_mat = [[conf_mat[1][1], conf_mat[1][0]], [conf_mat[0][1], conf_mat[0][0]]]
                tpr = float(conf_mat[0][0]) / (conf_mat[0][0] + conf_mat[0][1] + 1)
                fpr = float(conf_mat[1][0]) / (conf_mat[1][0] + conf_mat[1][1] + 1)
                precision = float(conf_mat[0][0]) / (conf_mat[0][0] + conf_mat[1][0] + 1)
                false_positive_rates.append(fpr)
                true_positive_rates.append(tpr)
                # for row in conf_mat:
                #     print row
                if (tpr > 0.7) & (fpr < 0.5):
                    print "Model: ", model
                    print "Cutoff: ", cutoff
                    print "Threshold: ", thresholds[i]
                    print "TPR/Recall: ", tpr
                    print "FPR: ", fpr
                    print "Precision: ", precision
                    print ""

            area = auc(false_positive_rates, true_positive_rates)

            ax.plot(
                false_positive_rates,
                true_positive_rates,
                label="ROC, model: {0} (area = {1:0.2f})".format(model, area)
            )
    ax.plot([0,1], [0, 1], linestyle="--", color="k")
    ax.scatter(0.2, 0.8, color="k", label="goal")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Cross-Validated ROC Curve")
    plt.legend(loc="lower right")
    plt.show()



if __name__ == '__main__':
    main()
