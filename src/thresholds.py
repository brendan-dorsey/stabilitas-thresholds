from stabilitasfilter import StabilitasFilter
from stabilitasfinder import StabilitasFinder
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
from datetime import datetime
import json
plt.style.use("ggplot")


def main():
    """
    Main function to run both layers of Stabilitas Thresholds app.
    """
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
        window_size="1w",
        anomaly_threshold=1,
        precalculated=True,
        quadratic=True,
        save_labels=False
    )

    anomalies_df = filter_layer.get_anomaly_reports(
        write_to_file=True,
        filename="data/2016/flagged_reports_quad_1std.csv"
        )
    date_lookup = filter_layer.date_lookup
    city_lookup = filter_layer.city_lookup

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


    finder_layer.label_critical_reports()
    # finder_layer.preprocesses_data(mode="evaluate")

    # finder_layer.fit()
    # finder_layer.predict()

    finder_layer.cross_val_predict()
    finder_layer._labeled_critical_cities_by_day()
    finder_layer._predicted_critical_cities_by_day()
    finder_layer._most_critical_report_per_city_per_day()

    finder_finish = time.time()

    # print finder_layer.date_lookup.keys()
    # print len(finder_layer.date_lookup["2016-12-20"])
    # print len(finder_layer.date_lookup["2016-12-20"][0])
    # print len(finder_layer.date_lookup["2016-12-20"][1])
    # print finder_layer.date_lookup["2016-12-20"][1]

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

    with open("app/date_lookup.json", mode="w") as f:
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

    with open("app/city_lookup.json", mode="w") as f:
        json.dump(city_lookup, f)


    ########################################
    ########################################
    ##                                    ##
    ##  This code will take lat/longs and ##
    ##   plot them as if on a map         ##
    ##                                    ##
    ########################################
    ########################################

    # lats, longs = filter_layer.get_anomaly_locations("2016-12-25")
    # print "Number of elevated cities: ", len(lats)
    # plt.scatter(longs, lats)
    # plt.show()


if __name__ == '__main__':
    main()
