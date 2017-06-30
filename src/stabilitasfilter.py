import pandas as pd
import numpy as np
from haversine import haversine

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

    def _load_cities(
            self,
            cities_filename,
            cleaned=True,
            min_size=300000
        ):
        """
        Load cities data from the given file.

        Input: filename
        Output: Self with stored city data.
        """
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

    def fit(
        self,
        data_filename,
        resample_size,
        window_size,
        anomaly_threshold
    ):
        """
        Fit model to reports included in file. This method loads the data,
        preprocesses it, builds time series, and identifies anomalies.

        Inputs:
            filename - str, path to report data
            resample_size - int, minutes for timeseries resampling
            window_size - str, length of window for rolling mean and
                standard deviation
                Valid inputs: "12h", "1d", "1w", "4w"
            anomaly_threshold - int, number of standard deviations above
                the mean a resample bucket needs to be in order to pass
                through the filter to the next layer
        Output: fit model ready to return anomaly information
        """
        self._load_data(data_filename)

    def _load_data(self, data_filename):
        """
        Load data from the given file.

        Input: filename
        Output: Self with stored and processed report data
        """
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


    def _preprocess_data(self):
        """
        Clean input data and build engineered features for use by the
        model.

        Input: self with raw dataframe initialized
        Output: self with cleaned and engineered dataframe
        """
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
        self.reports_df["severity_quadratic"] =\
            self.reports_df["severity"].map(severity_score_quadratic)

    def _build_cities(self):
        pass

    def find_anomalies(self, city):
        pass

    def test(self):
        print "Cities Loaded, shape: ", self.cities_df.shape
        print "Reports Loaded, shape: ", self.reports_df.shape
        print ""
        print self.reports_df.info()
        print "Skeleton working"

if __name__ == '__main__':
    main()
