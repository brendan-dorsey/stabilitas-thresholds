from stabilitasfilter import StabilitasFilter
from stabilitasfinder import StabilitasFinder
# import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
from datetime import datetime
import json
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, auc
from itertools import combinations, product
# plt.style.use("ggplot")


def main():
    """
    Function to test implementation of Stabilitas Finder.
    """
    window = "1wk"
    # model_type = "rfc"

    with open("debug/filter_full_date_lookup_{}.json".format(window)) as f:
        date_lookup = json.load(f)

    with open("debug/filter_full_city_lookup_{}.json".format(window)) as f:
        city_lookup = json.load(f)

    finder_start = time.time()
    finder_layer = StabilitasFinder()
    finder_layer.load_data(
        source="debug/flagged_reports_quad_{}_full.csv".format(window),
        date_lookup=date_lookup,
        city_lookup=city_lookup
    )

    finder_layer.label_critical_reports(cutoff=30)

    finder_layer.cross_val_predict()
    finder_layer._labeled_critical_cities_by_day()
    finder_layer._predicted_critical_cities_by_day()
    finder_layer._most_critical_report_per_city_per_day()

    finder_finish = time.time()

    print ""
    print "Finder finished at {0} in {1} seconds.".format(
                                    datetime.now().time(),
                                    finder_finish-finder_start
    )

    with open("debug/debug_full_finder_date_lookup_{}.json".format(window), mode="w") as f:
        json.dump(finder_layer.date_lookup, f)

    city_lookup = finder_layer.city_lookup

    drop_keys = ["timeseries", "anomalies"]
    for key in drop_keys:
        for sub_dict in city_lookup.values():
            if isinstance(sub_dict, dict):
                try:
                    del sub_dict[key]
                except KeyError:
                    pass

    with open("debug/debug_full_finder_city_lookup_{}.json".format(window), mode="w") as f:
        json.dump(city_lookup, f)

    y_true = finder_layer.flagged_df["critical"].values
    y_pred = finder_layer.flagged_df["predicted"].values

    conf_mat = finder_layer.confusion_matrix
    # Transpoition of sklearn confusion matrix to my preferred format:
    # TP  FN
    # FP  TN
    conf_mat = [
        [conf_mat[1][1], conf_mat[1][0]],
        [conf_mat[0][1], conf_mat[0][0]]
    ]

    # True Positive Rate: TP / TP + FN
    tpr = float(conf_mat[0][0]) / (conf_mat[0][0] + conf_mat[0][1] + 1)
    # False Positive Rate: FP / FP + TN
    fpr = float(conf_mat[1][0]) / (conf_mat[1][0] + conf_mat[1][1] + 1)
    # Precision: TP / TP + FP
    precision = float(conf_mat[0][0]) / (conf_mat[0][0] + conf_mat[1][0] + 1)
    # False Discovery Rate: FP / TP + FP
    fdr = float(conf_mat[1][0]) / (conf_mat[0][0] + conf_mat[1][0] + 1)
    if (precision + tpr) == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * tpr) / (precision + tpr)

    print "Confusion Matrix:"
    for row in conf_mat:
        print "     ", row

    print ""
    print "AUC: ", roc_auc_score(y_true, y_pred)
    print "Precision: ", precision
    print "True Positive Rate (Recall): ", tpr
    print "False Positive Rate: ", fpr
    print "False Discovery Rate: ", fdr
    print "F1 Score: ", f1






    ########################################
    ########################################
    ##                                    ##
    ##  The code below is for generating  ##
    ##  a ROC curve and determining an    ##
    ##  optimal decision threshold.       ##
    ##                                    ##
    ########################################
    ########################################

    # runs = []
    # for _ in range(1000):
    # thresholds = np.linspace(0, 1, 101)
    # thresholds = [0.36, 0.37, 0.38]
    # false_positive_rates = []
    # true_positive_rates = []
    # for threshold in thresholds:
    #     conf_mat = finder_layer.predict(threshold=threshold)
    #     fpr = float(conf_mat[0][1]) / (conf_mat[0][1] + (conf_mat[0][0]) + 1)
    #     tpr = float(conf_mat[1][1]) / (conf_mat[1][1] + (conf_mat[1][0]) + 1)
    #
    #     false_positive_rates.append(fpr)
    #     true_positive_rates.append(tpr)
    #
    # iteration = zip(thresholds, true_positive_rates, false_positive_rates)
    # print iteration

    # runs.append(iteration)
    # low, low_tpr, low_fpr, med, med_tpr, med_fpr, hi, hi_tpr, hi_fpr =\
    # 0, 0, 0, 0, 0, 0, 0, 0, 0
    # for run in runs:
    #     low += run[0][0]
    #     low_tpr += run[0][1]
    #     low_fpr += run[0][2]
    #     med += run[1][0]
    #     med_tpr += run[1][1]
    #     med_fpr += run[1][2]
    #     hi += run[2][0]
    #     hi_tpr += run[2][1]
    #     hi_fpr += run[2][2]
    # results = [
    #     (low/1000, low_tpr/1000, low_fpr/1000),
    #     (med/1000, med_tpr/1000, med_fpr/1000),
    #     (hi/1000, hi_tpr/1000, hi_fpr/1000)
    # ]
    # for result in results:
    #     print result
    # area = auc(false_positive_rates, true_positive_rates)
    # fig, ax = plt.subplots(1, figsize=(8,8))
    #
    # ax.plot(
    #     false_positive_rates,
    #     true_positive_rates,
    #     label="ROC curve (area = {:0.2f})".format(area)
    # )
    # ax.plot([0,1], [0, 1], linestyle="--", color="k")
    # ax.scatter(0.2, 0.99, color="k", label="goal")
    #
    # ax.set_xlabel("False Positive Rate")
    # ax.set_ylabel("True Positive Rate")
    # ax.set_xlim(0, 1)
    # ax.set_ylim(0, 1)
    # ax.set_title("ROC Curve for First Model")
    # plt.legend(loc="lower right")
    # plt.show()





if __name__ == '__main__':
    main()
