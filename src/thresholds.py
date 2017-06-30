from stabilitasfilter import StabilitasFilter
import matplotlib.pyplot as plt
import pandas as pd
import time
from datetime import datetime
plt.style.use("ggplot")

def main():
    start = time.time()
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
        precalculated=True,
        quadratic=True
    )
    finish = time.time()
    print "Finished at {0} in {1} seconds.".format(
                                    datetime.now().time(),
                                    finish-start
                                )

    lats, longs = filter_layer.get_anomaly_locations("2016-12-14")
    print "Number of elevated cities: ", len(lats)
    plt.scatter(longs, lats)
    plt.show()

    # filter_layer.test()
    # plt.scatter(
    #     filter_layer.cities_anomalies["Berlin"].index,
    #     filter_layer.cities_anomalies["Berlin"].values
    # )
    # plt.xlim("2016-12-12", "2016-12-27")

    # anomaly_counts = [len(cities) for cities in filter_layer.date_dictionary.values()]
    # plt.bar(filter_layer.date_dictionary.keys(), anomaly_counts, width=1)


if __name__ == '__main__':
    main()
