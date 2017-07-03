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
    finder.load_data("data/flagged_reports.csv")

    finder.label_critical_reports()

    # cutoffs = [10, 20, 30, 40, 50]
    # for cutoff in cutoffs:
    #     finder.label_critical_reports(cutoff)

    finder.fit()
    finder.predict_proba()
    # print finder.predict()

    thresholds = np.linspace(0.00, 1, 100)
    false_positive_rates = []
    true_positive_rates = []
    for threshold in thresholds:
        conf_mat = finder.predict(threshold=threshold)
        fpr = float(conf_mat[0][1]) / (conf_mat[0][1] + (conf_mat[0][0]) + 1)
        tpr = float(conf_mat[1][1]) / (conf_mat[1][1] + (conf_mat[1][0]) + 1)

        false_positive_rates.append(fpr)
        true_positive_rates.append(tpr)

    area = auc(false_positive_rates, true_positive_rates)
    fig, ax = plt.subplots(1, figsize=(8,8))

    ax.plot(
        false_positive_rates,
        true_positive_rates,
        label="ROC curve (area = {:0.2f})".format(area)
    )
    ax.plot([0,1], [0, 1], linestyle="--", color="k")
    ax.scatter(0.2, 0.99, color="k", label="goal")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("ROC Curve for First Model")
    plt.legend(loc="lower right")
    plt.show()





if __name__ == '__main__':
    main()
