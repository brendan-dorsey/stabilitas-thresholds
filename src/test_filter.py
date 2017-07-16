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
    Workspace to test filter layer of model.
    """
    filter_start = time.time()
    print "Started at {}.".format(datetime.now().time())
    # Filepaths assume running this script from stabilitas-thresholds/ dir
    cities_filename = "debug/cities300000.csv"
    filter_layer = StabilitasFilter(cities_filename, cleaned=True)

    data_filename = "debug/DEC_subset/reports_12DEC16-26DEC16.tsv"
    filter_layer.fit(
        data_filename,
        start_datetime="2016-12-12",
        end_datetime="2016-12-27",
        resample_size=3,
        window_size="1w",
        anomaly_threshold=1,
        load_city_labels=True,
        city_labels_path="debug/DEC_subset/DEC_city_labels.csv",
        quadratic=True,
        save_labels=False,
    )

    # filter_layer.reports_df = pd.read_csv("data/2016_flagged_reports_quad_1std_1wk.csv")



    # anomales_df = filter_layer.get_anomaly_reports(
    #     write_to_file=False,
    #     filename=None
    #     )i

    print len(anomalies_df["anomalous"])

    exit()

    # date_lookup = filter_layer.date_lookup
    # city_lookup = filter_layer.city_lookup
    #
    # filter_finish = time.time()
    # print "Filter finished at {0} in {1} seconds.".format(
    #                                 datetime.now().time(),
    #                                 filter_finish-filter_start
    #                             )
    #
    # with open("data/2016_quad_4w_date_lookup.json", mode="w") as f:
    #     json.dump(date_lookup, f)
    #
    # truncated_city_lookup = {}
    # for city in city_lookup.keys():
    #     truncated_city_lookup[city] = {
    #         "location": city_lookup[city]["location"]
    #     }
    # with open("data/2016_quad_4w_city_lookup.json", mode="w") as f:
    #     json.dump(truncated_city_lookup, f)


    # _load_cities
    # _build_cities_timeseries(quadratic=True)
    # _find_anomalies
    # _anomalies_by_day

    # cities_filename = "data/cities300000.csv"
    # filter_layer = StabilitasFilter(cities_filename, cleaned=True)
    # filter_layer._load_data("data/reports_12DEC16-26DEC16.tsv", precalculated=False, save_labels=False)
    # filter_layer.reports_df = pd.read_csv("data/2016_flagged_reports_quad_1std_1wk.csv")
    # filter_layer._build_cities_timeseries(quadratic=True)




if __name__ == '__main__':
    main()
