import pandas as pd
import numpy as np
from haversine import haversine
from itertools import izip
from collections import defaultdict
import time
import datetime
from multiprocessing import Process, Pool, cpu_count
from functools import partial
import random



class StabilitasFilter(object):
    def __init__(self, cities_filename, cleaned=True):
        """
        Instantiate a StabilitasFilter object with cities specified in
        the passed file.

        Callable methods are:
        fit - load and process data, identify anomalies, build reference
            dictionaries
        get_anomaly_reports - returns dataframe of anomalous reports to pass to
            StabilitasFinder model. Will write the results to a csv file in data
            directory by default (disable by calling with False as argument)
        get_anomaly_locations - returns tuple of lists of latitudes and
            longitudes of cities with anomalies detected on the passed in
            date -> ([lats], [longs])
        get_anomaly_cities - returns list of cities with anomalies detected on
            the passed in date, primarily for use within the
            get_anomaly_locations method -> [cities]
        """
        self._load_cities(cities_filename, cleaned)
        self.window_to_minutes_converter = {
            "12h": 720,
            "1d": 1440,
            "1w": 10080,
            "4w": 40320
        }

    def _load_cities(self, cities_filename, cleaned=True):
        """
        Load cities data from the given file.

        Input: filename
        Output: Self with stored city data.
        """
        start = time.time()
        print "Loading cities..."
        if cleaned:
            self.cities_df = pd.read_csv(cities_filename)
        else:
            city_columns = [
                "geonameid",
                "name",
                "asciiname",
                "alternatenames",
                "latitude",
                "longitude",
                "feature_class",
                "feature_code",
                "country_code",
                "cc2",
                "admin1_code",
                "admin2_code",
                "admin3_code",
                "admin4_code",
                "population",
                "elevation",
                "dem",
                "timezone",
                "modification_date"
            ]

            target_city_columns = [
                "name",
                "latitude",
                "longitude",
                "country_code"
            ]

            self.cities_df = pd.read_table(
                cities_filename,
                header=None,
                names=city_columns
            )
            self.cities_df = self.cities_df[
                            cities_df["population"] > min_size
                        ]
            self.cities_df = self.cities_df.reset_index()
            self.cities_df = self.cities_df[target_city_columns]

        self.cities_df["lat_long"] = zip(
                                self.cities_df["latitude"],
                                self.cities_df["longitude"]
                            )
        self.city_lookup = {}
        for city in self.cities_df["name"].unique():
            lat_long = self.cities_df[
                        self.cities_df["name"] == city
                        ]["lat_long"].values[0]
            self.city_lookup[city] = {"location": lat_long}
        finish = time.time()
        print "{0} cities loaded in {1} seconds.".format(
            len(self.cities_df), finish-start
        )

    def fit(
        self,
        data_filename,
        start_datetime,
        end_datetime,
        resample_size=3,
        window_size="1w",
        anomaly_threshold=1,
        load_city_labels=False,
        city_labels_path=None,
        quadratic=True,
        save_labels=False
    ):
        """
        Fit model to reports included in file. This method loads the data,
        preprocesses it, builds time series, and identifies anomalies.

        Inputs:
            filename - str, path to report data
            start_date - str, format "YYYY-MM-DD [HH:MM:SS] <-optional"
            end_date - str, format "YYYY-MM-DD [HH:MM:SS] <-optional"
            resample_size - int, minutes for timeseries resampling
                Default: 3 minutes
            window_size - str, length of window for rolling mean and standard
                deviation
                Valid inputs: "12h", "1d", "1w", "4w"
                Default: 1 week
            anomaly_threshold - int, number of standard deviations above
                the mean a resample bucket needs to be in order to pass
                through the filter to the next layer
                Default: 1 standard deviation
            load_city_labels - bool, indicate whether city labels have been
                precalculated or not
                Default: False
            quadratic - bool, indicate whether to use engineered severity
                score or reporting volumes
                Default: True
        Output: fit model ready to return anomaly information
        """
        self.resample_size = resample_size
        self.time_delta = pd.Timedelta(minutes=self.resample_size)
        self.window = \
            self.window_to_minutes_converter[window_size] / resample_size
        self.threshold = anomaly_threshold
        self.start = pd.to_datetime(start_datetime)
        self.end = pd.to_datetime(end_datetime)
        self.date_range = pd.date_range(self.start, self.end)

        self._load_data(data_filename, load_city_labels, city_labels_path, save_labels)
        self._build_cities_timeseries(quadratic)
        self._find_anomalies()
        self._anomalies_by_day()

    def _load_data(self, data_filename, load_city_labels, city_labels_path, save_labels):
        """
        Load data from the given file.

        Input: filename
        Output: Self with stored and processed report data
        """
        print "Loading data..."
        start = time.time()
        column_names = [
            "lat",
            "lon",
            "id",
            "source_id",
            "account_id",
            "title",
            "created_on",
            "updated_on",
            "start_ts",
            "until_ts",
            "report_type",
            "notes",
            "layer_id",
            "severity"
        ]

        self.reports_df = pd.read_table(
            data_filename,
            header=None,
            names=column_names
        )
        self._preprocess_data()
        self._map_reports_to_cities(load_city_labels, city_labels_path, save_labels)
        self._build_multiindex()
        finish = time.time()
        print "{0} reports loaded in {1} seconds.".format(
            len(self.reports_df), finish-start
        )

    def _preprocess_data(self):
        """
        Clean input data and build engineered features for use by the
        model.

        Input: self with raw dataframe initialized
        Output: self with cleaned and engineered dataframe
        """
        print "        Processing data..."
        start = time.time()


        target_columns = [
            "lat",
            "lon",
            "id",
            "title",
            "start_ts",
            "report_type",
            "severity"
        ]

        self.reports_df.dropna(axis=0, how="any", inplace=True)
        self.reports_df = self.reports_df[target_columns]
        self.reports_df["start_ts"] = pd.to_datetime(
                                            self.reports_df["start_ts"],
                                            unit="s",
                                            errors="ignore"
                                        )
        self.reports_df = self.reports_df[
                            self.reports_df["start_ts"] > self.start
                        ]
        self.reports_df = self.reports_df[
                            self.reports_df["start_ts"] < self.end
                        ]

        self.reports_df["lat_long"] = zip(
            self.reports_df["lat"],
            self.reports_df["lon"]
        )
        self.reports_df["severity_quadratic"] =\
            self.reports_df["severity"].map(severity_score_quadratic)

        finish = time.time()
        print "        Reports processed in {0} seconds".format(finish-start)

    def _map_reports_to_cities(self, load_city_labels=False, city_labels_path=None, save_labels=False):
        """
        Method that calculates the haversine distance from each report to each
        city, identifies the city closest to each report, and labels each
        report with its closest city.

        Inputs: load_city_labels - bool, indicates whether to use load_city_labels
        values.
        Output: self with labeled reports dataframe
        """
        print "        Labeling reports with cities..."
        start = time.time()

        if load_city_labels:
            city_labels = pd.read_csv(city_labels_path, header=None)
            city_labels = city_labels[1].values

        else:
            city_labels = []
            city_label_indices = []
            total_reports = len(self.reports_df)

            for i, report in enumerate(self.reports_df["lat_long"]):
                if i % 1000 == 0:
                    current_time = time.time() - start
                    total_time = (current_time * total_reports) / (i+1)
                    print "     Estimated {0} seconds remaining for {1} reports".format(
                        int(round(total_time - current_time)),
                        total_reports - i
                    )
                distances = []
                for city in self.cities_df["lat_long"]:
                    distances.append(haversine(report, city))
                city_label_indices.append(np.argmin(distances))

            for index in city_label_indices:
                city_labels.append(self.cities_df.loc[index, "name"])

        if save_labels:
            labels = pd.DataFrame(city_labels)
            labels.to_csv(
                "debug/DEC_city_labels.csv",
                header=False,
                mode="w"
            )

        self.reports_df["city"] = city_labels

        finish = time.time()
        print "        Reports labeled with cities in {0} seconds.".format(
            finish-start
        )

    def _build_multiindex(self):
        """
        Method to build multi-index for increased performance.
        """
        column_names = [
            "lat",
            "lon",
            "id",
            "title",
            "start_ts",
            "report_type",
            "severity",
            "lat_long",
            "severity_quadratic",
            "city"
        ]
        arrays = [
            pd.to_datetime(self.reports_df["start_ts"].values),
            self.reports_df["city"].values,
            range(len(self.reports_df))
        ]
        multi_index = pd.MultiIndex.from_arrays(arrays, names=["time", "city", "row"])

        self.reports_df = pd.DataFrame(
            self.reports_df.values,
            index=multi_index,
            columns=column_names
            )
        self.reports_df.sort_index(inplace=True)
        self.reports_df["start_ts"] =\
            pd.to_datetime(self.reports_df["start_ts"])
        self.reports_df["severity_quadratic"] =\
            self.reports_df["severity_quadratic"].apply(float)


    def _build_cities_timeseries(self, quadratic):
        """
        Method to build resampled timeseries for each city in the dataset.
        Inputs: quadratic - bool, indicate whether to use quadratic severity
            score (True) or volume of reports (False)
        """
        print "Building city timeseries..."
        start = time.time()
        idx = pd.IndexSlice

        for city in set(self.reports_df.index.get_level_values("city")):
            city_df = self.reports_df.loc[idx[:, city, :], idx[:]]

            if quadratic:
                # Use engineered quadratic severity score
                ts = pd.Series(
                    city_df["severity_quadratic"].values,
                    index=city_df["start_ts"]
                )
                self.city_lookup[city]["timeseries"] = np.log(
                    ts.resample("{}T".format(self.resample_size)
                    ).sum())

                # Try with mean instead of sum
                # self.city_lookup[city]["timeseries"] = np.log(
                #     ts.resample("{}T".format(self.resample_size)).mean()
                # )

            else:
                # Use only volume of reporting
                ts = pd.Series(
                    np.ones(len(city_df)),
                    index=city_df["start_ts"]
                )
                self.city_lookup[city]["timeseries"] = \
                    ts.resample("{}T".format(self.resample_size)).sum()

        finish = time.time()
        print "City timeseries calculated in {0} seconds.".format(finish-start)

    def _find_anomalies(self):
        """
        Method to find anomalies in the built timeseries.
        """
        print "Detecting anomalies..."
        start = time.time()
        idx = pd.IndexSlice
        total_cities = len(self.reports_df["city"].unique())
        self.reports_df["anomalous"] = np.zeros(len(self.reports_df))

        for i, city in enumerate(self.reports_df["city"].unique()):
            if i % 10 == 0:
                current_time = time.time() - start
                total_time = (current_time * total_cities) / (i+1)
                print "     Estimated {0} seconds remaining for {1} cities".format(
                    int(round(total_time - current_time)),
                    total_cities - i
                )
            series = self.city_lookup[city]["timeseries"]

            rolling_std = series.rolling(
                window=self.window,
                min_periods=1,
                center=False
            ).std()
            rolling_mean = series.rolling(
                window=self.window,
                min_periods=1,
                center=False
            ).mean()

            threshold = rolling_mean + self.threshold * rolling_std
            locations = range(len(threshold))

            anomalies = []
            self.city_lookup[city]["anomaly_indices"] = []
            for sample, threshold, location in izip(series, threshold, locations):
                if sample > threshold:
                    anomalies.append(sample)
                    window_start = (series.index[location] - self.time_delta)
                    window_stop = series.index[location]
                    self.city_lookup[city]["anomaly_indices"].append((window_start, window_stop))
                    # self.reports_df.loc[
                    #     idx[window_start:window_stop, city, :],
                    #     idx["anomalous"]
                    # ] = 1
                else:
                    anomalies.append(None)

            anomalies = pd.Series(anomalies, index=series.index)
            self.city_lookup[city]["anomalies"] = anomalies.dropna()

        print sum(self.reports_df["anomalous"])
        finish = time.time()
        print "Anomalies detected in {0} seconds.".format(finish-start)

    def _anomalies_by_day(self):
        """
        Method to convert identified anomalies into date:[anomalous_cities]
        pairs.
        """
        print "Grouping anomalous cities by day..."
        start = time.time()
        self.date_lookup = defaultdict(list)
        for city in self.city_lookup.keys():
            try:
                series = self.city_lookup[city]["anomalies"]
            except:
                continue
            daily_anomalies = series.resample("d").sum()[self.start:self.end]
            for day in daily_anomalies.index:
                key = str(day.date())
                if daily_anomalies[day] > 0:
                    if len(self.date_lookup[key]) == 0:
                        self.date_lookup[key].append([city])
                    else:
                        self.date_lookup[key][0].append(city)

        finish = time.time()
        print "Anomalies grouped in {0} seconds.".format(finish-start)

    def get_anomaly_cities(self, date):
        """
        Method to return all the anomalous cities on a given day.
        Input: date - str, format "YYYY-MM-DD"
        Outpus: list of cities
        """
        query = pd.to_datetime(date).date()
        return self.date_lookup[query]

    def get_anomaly_locations(self, date):
        """
        Method to return location data for anomalous cities on the given date.
        Input: date - str, format "YYYY-MM-DD"
        Output: tuple of two lists ([latitudes], [longitudes])
        """
        cities = self.get_anomaly_cities(date)
        lats = [self.city_lookup[city]["location"][0] for city in cities]
        longs = [self.city_lookup[city]["location"][1] for city in cities]

        return (lats, longs)

    def get_anomaly_reports(
        self,
        write_to_file=True,
        filename="data/flagged_reports.csv"
    ):
        """
        Method to apply boolean flag to reports to indicate whether they are
        part of anomalous time buckets.
        """
        print "Flagging anomalous reports..."
        start = time.time()
        self.reports_df["anomalous"] = np.zeros(len(self.reports_df))
        time_delta = pd.Timedelta(minutes=self.resample_size)
        idx = pd.IndexSlice
        total_cities = len(self.city_lookup.keys())

        for i, city in enumerate(self.city_lookup.keys()):
            if i % 10 == 0:
                current_time = time.time() - start
                total_time = (current_time * total_cities) / (i+1)
                print "     Estimated {0} seconds remaining for {1} cities".format(
                    int(round(total_time - current_time)),
                    total_cities - i
                )
            try:
                anomalies = self.city_lookup[city]["anomaly_indices"]
                for window_start, timestamp in anomalies:
                    # window_start = timestamp - time_delta
                    self.reports_df.loc[
                        idx[window_start:timestamp, city, :],
                        idx["anomalous"]
                    ] = 1
            except KeyError:
                pass

        #### Multiprocessing attempt ####
        # cities = self.reports_df["city"].unique()
        # self.reports_df = pooled_labeling(
        #     idx,
        #     time_delta,
        #     self.city_lookup,
        #     self.reports_df,
        #     cities
        # )



        anomalies_df = self.reports_df[self.reports_df["anomalous"] == 1]
        if write_to_file:
            anomalies_df.to_csv(
                filename,
                mode="w"
            )
        finish = time.time()
        print "{0} anomalies flagged in {1} seconds.".format(
            sum(self.reports_df["anomalous"]), finish-start
        )
        return anomalies_df


def severity_score_quadratic(severity_rating):
    if severity_rating == "low":
        return 1
    elif severity_rating == "moderate":
        return 4
    elif severity_rating == "medium":
        return 9
    elif severity_rating == "high":
        return 16
    elif severity_rating == "extreme":
        return 25
    else:
        return 4

def label_anomalous_reports(
    idx,
    time_delta,
    dictionary,
    dataframe,
    city
):
    print "Worker checking:      {}".format(city)
    start = time.time()
    try:
        anomalies = dictionary[city]["anomalies"]
        for timestamp in anomalies.index:
            window_start = timestamp - time_delta
            dataframe.loc[
                idx[window_start:timestamp, city, :],
                idx["anomalous"]
            ] = 1
    except KeyError:
        pass
    print "Worker finished in {} seconds.".format(time.time() - start)
    return dataframe.loc[idx[:, city, :], idx[:]]

def pooled_labeling(
    idx,
    time_delta,
    dictionary,
    dataframe,
    cities,
):
    start = time.time()

    pool = Pool(cpu_count())
    function = partial(
        label_anomalous_reports,
        idx,
        time_delta,
        dictionary,
        dataframe,
    )
    print "     Mapping pool..."
    dataframe = pd.concat(pool.map(function, cities))
    pool.close()
    pool.join()
    return dataframe
