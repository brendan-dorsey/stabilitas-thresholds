from stabilitasfilter import StabilitasFilter
import matplotlib.pyplot as plt
import pandas as pd
import time
from datetime import datetime
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

    data_filename = "data/reports_12DEC16-26DEC16.tsv"
    filter_layer.fit(
        data_filename,
        start_datetime="2016-12-12",
        end_datetime="2016-12-26",
        resample_size=3,
        window_size="1w",
        anomaly_threshold=1,
        precalculated=False,
        quadratic=True
    )

    # These settings return 8.4% of reports as anomalous out of the sample data
    # Layer completes in about 330 seconds for 92884 reports.
    # This includes haversine calculations for each city/report combination.
    # Layer can complete ~280 reports per second

    anomalies_df = filter_layer.get_anomaly_reports(write_to_file=True)
    date_lookup = filter_layer.date_lookup
    city_lookup = filter_layer.city_lookup

    filter_finish = time.time()
    print "Filter finished at {0} in {1} seconds.".format(
                                    datetime.now().time(),
                                    finish-start
                                )

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
