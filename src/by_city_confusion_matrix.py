import numpy as np
import pandas as pd
import json
from sklearn.metrics import confusion_matrix
from itertools import combinations, product


def main():
    with open("app/date_lookup.json", "r") as f:
        date_lookup = json.load(f)

    dates = date_lookup.keys()

    cities = set()
    for date in dates:
        for city in date_lookup[date][0]:
            cities.add(city)

    city_date_pairs = product(cities, dates)

    y_true = []
    y_pred = []
    for city, date in city_date_pairs:
        if city in date_lookup[date][1]:
            y_true.append(1)
        else:
            y_true.append(0)
        if city in date_lookup[date][2]:
            y_pred.append(1)
        else:
            y_pred.append(0)

    conf_mat = confusion_matrix(y_true, y_pred)
    # Transpoition of sklearn confusion matrix to this format:
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
        print row

    print "True Positive Rate (Recall): ", tpr
    print "False Positive Rate: ", fpr
    print "False Discovery Rate: ", fdr
    print "Precision: ", precision
    print "F1 Score: ", f1





if __name__ == '__main__':
    main()
