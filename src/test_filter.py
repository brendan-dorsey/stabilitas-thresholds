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
        start_datetime="2016-10-01",
        end_datetime="2017-12-01",
        resample_size=3,
        window_size="1w",
        anomaly_threshold=1,
        precalculated=False,
        quadratic=False,
        save_labels=False
    )

    anomalies_df = filter_layer.get_anomaly_reports(
        write_to_file=True,
        filename="data/OCT_flagged_reports_vol_1std.csv"
        )

    date_lookup = filter_layer.date_lookup
    city_lookup = filter_layer.city_lookup

    filter_finish = time.time()
    print "Filter finished at {0} in {1} seconds.".format(
                                    datetime.now().time(),
                                    filter_finish-filter_start
                                )

    # with open("app/date_lookup.json", mode="w") as f:
    #     json.dump(finder_layer.date_lookup, f)
    #
    # truncated_city_lookup = {}
    # for city in finder_layer.city_lookup.keys():
    #     truncated_city_lookup[city] = {
    #         "location": finder_layer.city_lookup[city]["location"]
    #     }
    # with open("app/city_lookup.json", mode="w") as f:
    #     json.dump(truncated_city_lookup, f)



if __name__ == '__main__':
    main()
