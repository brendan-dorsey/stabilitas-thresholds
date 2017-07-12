from stabilitasfinder import StabilitasFinder
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
from datetime import datetime
from sklearn.metrics import auc
plt.style.use("ggplot")


def main():
    """
    Function to test implementation of Stabilitas Finder.
    """
    finder = StabilitasFinder()
    finder.load_data(source="data/OCT_flagged_reports_vol_1std.csv")

    finder.label_critical_reports()
    finder.cross_val_predict()
    # finder.extract_critical_titles()





    # conf_mat = finder.confusion_matrix
    # Transpoition of sklearn confusion matrix to my preferred format:
    # TP  FN
    # FP  TN
    # conf_mat = [[conf_mat[1][1], conf_mat[1][0]], [conf_mat[0][1], conf_mat[0][0]]]
    #
    # for row in conf_mat:
    #     print row

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
    #     conf_mat = finder.predict(threshold=threshold)
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
