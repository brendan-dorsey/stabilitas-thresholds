import numpy as np
import pandas as pd
import json
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
from itertools import combinations, product
import matplotlib.pyplot as plt
plt.style.use("ggplot")


def main():
    """
    Main function to run script.
    """

    filename = "app/date_lookup.json"
    by_city_confusion_matrix(filename)


def by_city_confusion_matrix(filename):
    """
    Function to generate a confusion matrix and several performance
    metrics on a per-city/per-day basis. Input is file path for the date
    lookup dictionary from the final output of the model. Output is printed to
    the command line.

    Inputs:
    filename = str, project path to target final date lookup dictionary.
    """
    with open(filename, "r") as f:
        date_lookup = json.load(f)

    dates = date_lookup.keys()
    print len(dates)

    cities = set()
    for date in dates:
        try:
            for city in date_lookup[date][0]:
                cities.add(city)
        except IndexError:
            # print date
            # print date_lookup[date]
            continue

    city_date_pairs = product(cities, dates)

    y_true = []
    y_pred = []
    for city, date in city_date_pairs:
        try:
            if city in date_lookup[date][1]:
                y_true.append(1)
            else:
                y_true.append(0)
            if city in date_lookup[date][2]:
                y_pred.append(1)
            else:
                y_pred.append(0)
        except:
            # print date
            # print date_lookup[date]
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
    tpr = float(conf_mat[0][0]) / (conf_mat[0][0] + conf_mat[0][1])
    # False Positive Rate: FP / FP + TN
    fpr = float(conf_mat[1][0]) / (conf_mat[1][0] + conf_mat[1][1])
    # Precision: TP / TP + FP
    precision = float(conf_mat[0][0]) / (conf_mat[0][0] + conf_mat[1][0])
    # False Discovery Rate: FP / TP + FP
    fdr = float(conf_mat[1][0]) / (conf_mat[0][0] + conf_mat[1][0])
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

    # t, f, th = roc_curve(y_true, y_pred)
    # plt.plot(t, f, label="ROC")
    # plt.legend()
    # plt.show()

    ########################
    #    11 JUL RESULTS    #
    ########################

    # Confusion Matrix:
    #   [57, 4]
    #   [531, 8978]
    #
    # AUC:  0.939292197728
    # Precision:  0.0967741935484
    # True Positive Rate (Recall):  0.91935483871
    # False Positive Rate:  0.0558359621451
    # False Discovery Rate:  0.901528013582
    # F1 Score:  0.175115207373

    #################################
    #        12 JUL RESULTS         #
    #    first with full dataset    #
    #################################
    # Note: this model used VOLUME scoring only, not severity scoring

    # Confusion Matrix:
    #   [1178, 21]
    #   [17745, 329996]
    #
    # AUC:  0.965728023224
    # Precision:  0.0622489959839
    # True Positive Rate (Recall):  0.981666666667
    # False Positive Rate:  0.0510292113118
    # False Discovery Rate:  0.937698161065
    # F1 Score:  0.11707414033

if __name__ == '__main__':
    main()
