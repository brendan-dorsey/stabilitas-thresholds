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
    Workspace function to test filter layer.
    """
    # filter_start = time.time()
    # print "Started at {}.".format(datetime.now().time())
    # Filepaths assume running this script from stabilitas-thresholds/ dir
    # cities_filename = "data/cities300000.csv"
    # filter_layer = StabilitasFilter(cities_filename, cleaned=True)
    #
    # data_filename = "data/reports_12DEC16-26DEC16.tsv"
    # filter_layer.fit(
    #     data_filename,
    #     start_datetime="2016-12-12",
    #     end_datetime="2016-12-13",
    #     resample_size=3,
    #     window_size="1w",
    #     anomaly_threshold=1,
    #     precalculated=True,
    #     quadratic=True,
    #     write_to_file=True
    # )

    # These settings return 8.4% of reports as anomalous out of the sample data
    # Layer completes in about 330 seconds for 92884 reports.
    # This includes haversine calculations for each city/report combination.
    # Layer can complete ~280 reports per second

    # anomalies_df = filter_layer.get_anomaly_reports(write_to_file=False)
    # print anomalies_df.info()
    # print anomalies_df.describe()
    # return ""
    # date_lookup = filter_layer.date_lookup
    # city_lookup = filter_layer.city_lookup

    # filter_finish = time.time()
    # print "Filter finished at {0} in {1} seconds.".format(
    #                                 datetime.now().time(),
    #                                 filter_finish-filter_start
    #                             )

    df = pd.read_csv("data/test_multiindex_df.csv")
    df["time"] = pd.to_datetime(df["time"])
    arrays = [
        df["time"].values,
        df["city"].values,
        range(len(df))
    ]
    multi_index = pd.MultiIndex.from_arrays(arrays, names=["time", "city", "row"])
    df = pd.DataFrame(df.values, index=multi_index)
    df.sort_index(level="time", inplace=True)

    timestamp = pd.to_datetime("2016-12-12 18:17:23")
    time_delta = pd.Timedelta(minutes=3)
    start = timestamp - time_delta

    idx = pd.IndexSlice

    # print df.head()
    print df.loc[idx[start:timestamp, "Auckland", :], idx[:]]




        # ((self.reports_df["city"] == city) &
        # (self.reports_df["start_ts"] >= timestamp - time_delta) &
        # (self.reports_df["start_ts"] <= timestamp)),
        # "anomalous"
        # ] = 1

if __name__ == '__main__':
    main()
