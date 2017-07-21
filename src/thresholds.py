from stabilitasfilter import StabilitasFilter
from stabilitasfinder import StabilitasFinder
# import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
from datetime import datetime
import json
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
from itertools import combinations, product


def main():
    """
    Main function to run both layers of Stabilitas Thresholds app.
    """
    window_size = "1w"
    cutoff = 30
    filter_start = time.time()
    print "Started at {}.".format(datetime.now().time())
    # Filepaths assume running this script from stabilitas-thresholds/ dir
    cities_filename = "data/cities300000.csv"
    filter_layer = StabilitasFilter(cities_filename, cleaned=True)

    data_filename = "data/2016/all_2016.txt"
    filter_layer.fit(
        data_filename,
        start_datetime="2016-01-01",
        end_datetime="2017-01-01",
        resample_size=3,
        window_size=window_size,
        anomaly_threshold=1,
        load_city_labels=True,
        city_labels_path="data/outputs_2016/2016_city_labels.csv",
        quadratic=False,
        save_labels=False,
    )

    anomalies_df = filter_layer.get_anomaly_reports(
        write_to_file=True,
        filename="data/outputs_2016/flagged_reports_vol_{}_full.csv".format(
            window_size
        ))
    date_lookup = filter_layer.date_lookup
    city_lookup = filter_layer.city_lookup

    drop_keys = ["timeseries", "anomalies", "anomaly_indices"]
    for key in drop_keys:
        for sub_dict in city_lookup.values():
            if isinstance(sub_dict, dict):
                try:
                    del sub_dict[key]
                except KeyError:
                    pass

    with open("""data/outputs_2016/
    filter_full_vol_date_lookup_{}.json""".format(window_size), mode="w") as f:
        json.dump(date_lookup, f)

    with open("""data/outputs_2016/
    filter_full_vol_city_lookup_{}.json""".format(window_size), mode="w") as f:
        json.dump(city_lookup, f)

    filter_finish = time.time()
    print "Filter finished at {0} in {1} seconds.".format(
                                    datetime.now().time(),
                                    filter_finish-filter_start
                                )

    finder_start = time.time()
    finder_layer = StabilitasFinder()
    finder_layer.load_data(
        source=anomalies_df,
        date_lookup=date_lookup,
        city_lookup=city_lookup
    )

    finder_layer.label_critical_reports(cutoff=cutoff)

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
    print ""
    print "Ensemble finished at {0} in {1} seconds.".format(
                                    datetime.now().time(),
                                    finder_finish-filter_start
    )

    with open("""data/outputs_2016/
    full_final_vol_date_lookup_{}.json""".format(window_size), mode="w") as f:
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

    with open("""data/outputs_2016/
    full_final_vol_city_lookup_{}.json""".format(window_size), mode="w") as f:
        json.dump(city_lookup, f)

    y_true = finder_layer.flagged_df["critical"].values
    y_pred = finder_layer.flagged_df["predicted"].values

    conf_mat = finder_layer.confusion_matrix
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
        print "     ", row

    print ""
    print "AUC: ", roc_auc_score(y_true, y_pred)
    print "Precision: ", precision
    print "True Positive Rate (Recall): ", tpr
    print "False Positive Rate: ", fpr
    print "False Discovery Rate: ", fdr
    print "F1 Score: ", f1
    print ""

    ########################################
    ########################################
    #                                      #
    #   Code for by city by day metrics    #
    #                                      #
    ########################################
    ########################################

    date_lookup = finder_layer.date_lookup
    dates = date_lookup.keys()

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

    ########################################
    ########################################
    #                                      #
    #   This code will take lat/longs and  #
    #    plot them as if on a map          #
    #                                      #
    ########################################
    ########################################

    # lats, longs = filter_layer.get_anomaly_locations("2016-12-25")
    # print "Number of elevated cities: ", len(lats)
    # plt.scatter(longs, lats)
    # plt.show()


if __name__ == '__main__':
    main()
