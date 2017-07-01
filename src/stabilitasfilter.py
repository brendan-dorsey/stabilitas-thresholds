import pandas as pd
import numpy as np
from haversine import haversine
from itertools import izip
from collections import defaultdict


def main():
    print "Building Filter Model..."


class StabilitasFilter(object):
    def __init__(self, cities_filename, cleaned=True):
        """
        Instantiate a StabilitasFilter object with cities specified in
        the passed file.
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

    def fit(
        self,
        data_filename,
        start_datetime,
        end_datetime,
        resample_size=3,
        window_size="1w",
        anomaly_threshold=1,
        precalculated=False,
        quadratic=True
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
            precalculated - bool, indicate whether city labels have been
                precalculated or not
                Default: False
            quadratic - bool, indicate whether to use engineered severity
                score or reporting volumes
                Default: True
        Output: fit model ready to return anomaly information
        """
        self.resample_size = resample_size
        self.window = \
            self.window_to_minutes_converter[window_size] / resample_size
        self.threshold = anomaly_threshold
        self.start = pd.to_datetime(start_datetime)
        self.end = pd.to_datetime(end_datetime)
        self.date_range = pd.date_range(self.start, self.end)

        self._load_data(data_filename, precalculated)
        self._build_cities_timeseries(quadratic)
        self._find_anomalies()
        self._anomalies_by_day()

    def _load_data(self, data_filename, precalculated):
        """
        Load data from the given file.

        Input: filename
        Output: Self with stored and processed report data
        """
        print "Loading data..."
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
        self._map_reports_to_cities(precalculated)

    def _preprocess_data(self):
        """
        Clean input data and build engineered features for use by the
        model.

        Input: self with raw dataframe initialized
        Output: self with cleaned and engineered dataframe
        """
        print "Processing data..."

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

    def _map_reports_to_cities(self, precalculated=False):
        """
        Method that calculates the haversine distance from each report to each
        city, identifies the city closest to each report, and labels each
        report with its closest city.

        Inputs: precalculated - bool, indicates whether to use precalculated
        values.
        Output: self with labeled reports dataframe
        """
        print "Labeling reports with cities..."
        if precalculated:
            precalculated_filename = "data/city_label_indices.csv"
            city_label_indices = pd.read_csv(
                precalculated_filename,
                header=None
            )
            city_label_indices = city_label_indices[1].values
        else:
            city_label_indices = []
            for report in self.reports_df["lat_long"]:
                distances = []
                for city in self.cities_df["lat_long"]:
                    distances.append(haversine(report, city))
                city_label_indices.append(np.argmin(distances))
            indices_df = pd.DataFrame(city_label_indices)
            indices_df.to_csv(
                "data/city_label_indices.csv",
                header=False,
                mode="a"
            )

        city_labels = []
        for index in city_label_indices:
            city_labels.append(self.cities_df.ix[index, "name"])

        self.reports_df["city"] = city_labels

    def _build_cities_timeseries(self, quadratic):
        """
        Method to build resampled timeseries for each city in the dataset.
        """
        print "Building city timeseries..."
        # self.cities_timeseries = {}
        for city in self.reports_df["city"].unique():
            city_df = self.reports_df[self.reports_df["city"] == city]

            if quadratic:
                # Use engineered quadratic severity score
                ts = pd.Series(
                    city_df["severity_quadratic"].values,
                    index=city_df["start_ts"]
                )
                # self.city_lookup[city]["timeseries"] = np.log(
                #     ts.resample("{}T".format(self.resample_size)
                #     ).sum())

                # Try with mean instead of sum
                self.city_lookup[city]["timeseries"] = np.log(
                    ts.resample("{}T".format(self.resample_size)).mean()
                )
            else:
                # Use only volume of reporting
                ts = pd.Series(
                    np.ones(len(city_df)),
                    index=city_df["start_ts"]
                )
                self.city_lookup[city]["timeseries"] = \
                    ts.resample("{}T".format(self.resample_size)).sum()

    def _find_anomalies(self):
        """
        Method to find anomalies in the built timeseries.
        """
        print "Detecting anomalies..."
        # self.cities_anomalies = {}
        for city in self.reports_df["city"].unique():
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
            # print threshold
            # break

            anomalies = []
            for sample, threshold in izip(series, threshold):
                if sample > threshold:
                    anomalies.append(sample)
                else:
                    anomalies.append(None)

            # self.cities_anomalies[city] = pd.Series(
            #                                 anomalies,
            #                                 index=series.index
            #                             )
            anomalies = pd.Series(anomalies, index=series.index)
            self.city_lookup[city]["anomalies"] = anomalies.dropna()

    def _anomalies_by_day(self):
        """
        Method to convert identified anomalies into date:[anomalous_cities]
        pairs.
        """
        print "Grouping anomalous cities by day..."
        self.date_dictionary = defaultdict(list)
        for city in self.city_lookup.keys():
            try:
                series = self.city_lookup[city]["anomalies"]
            except:
                continue
            daily_anomalies = series.resample("d").sum()[self.start:self.end]
            # print daily_anomalies
            for day in daily_anomalies.index:
                # print daily_anomalies[day]
                if daily_anomalies[day] > 0:
                    self.date_dictionary[day.date()].append(city)

    def get_anomaly_cities(self, date):
        """
        Method to return all the anomalous cities on a given day.
        Input: date - str, format "YYYY-MM-DD"
        Outpus: list of cities
        """
        query = pd.to_datetime(date).date()
        return self.date_dictionary[query]

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

    def _flag_anomalous_reports(self):
        self.reports_df["anomalous"] = np.zeros(len(self.reports_df))
        time_delta = pd.Timedelta(minutes=self.resample_size)
        for city in self.city_lookup.keys():
            try:
                anomalies = self.city_lookup[city]["anomalies"]

                # if sum(anomalies) == 0:
                #     continue
                # print anomalies
                # print sum(anomalies)
                # break

                for time in anomalies.index:
                    print "City: ", city
                    print "Time: ", time
                    print "Start: ", time - time_delta
                    print self.reports_df[
                        # (self.reports_df["city"] == city) &
                        (self.reports_df["start_ts"] >= time - time_delta) &
                        (self.reports_df["start_ts"] <= time)
                    ]
                    break

                    # self.reports_df[
                        # (self.reports_df["city"] == city) &
                        # (self.reports_df["start_ts"] >= time - time_delta) &
                        # (self.reports_df["start_ts"] < time),
                        # "anomalous"
                        # ] += 1

                # if len(anomalies) > 5:
                #     print anomalies.index[0]
                #     print pd.Timedelta(minutes=self.resample_size)
                #     print anomalies.index[0] - \
                #     pd.Timedelta(minutes=self.resample_size)
                #     break
            except:
                continue
        print sum(self.reports_df["anomalous"])

    def test(self):
        print "Cities Loaded, shape: ", self.cities_df.shape
        print "Reports Loaded, shape: ", self.reports_df.shape
        print "Reports Mapped, cities: ", len(self.reports_df["city"].unique())
        print ""
        # print self.reports_df.info()
        print "Filter Functioning"

if __name__ == '__main__':
    main()
