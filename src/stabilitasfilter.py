import pandas as pd
import numpy as np
from haversine import haversine

def main():
    print "Building Filter Model..."

class StabilitasFilter(object):
    def __init__(self, cities_filename, cleaned=True):
        """
        Instantiate a StabilitasFilter object with cities specified in the
        passed file.
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

    def fit(self,
            filename,
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
        Output: fit model ready to return anomaly information
        """
        pass

    def _load_data(self, filename):
        """
        Load data from the given file.

        Input: filename
        Output: Self with stored and processed report data
        """
        pass

    def _preprocess_data(self, DataFrame):
        """
        Clean input data and build engineered features for use by the
        model.

        Input: Raw dataframe
        Output: Cleaned and engineered dataframe.
        """
        pass


    def _build_cities(self):
        pass

    def find_anomalies(self, city):
        pass

    def test(self):
        print "Cities Loaded, shape: ", self.cities_df.shape
        print "Skeleton working"

if __name__ == '__main__':
    main()
