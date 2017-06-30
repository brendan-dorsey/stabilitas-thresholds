import pandas as pd
import numpy as np

def main():
    print "Building Filter Model..."

class StabilitasFilter(object):
    def __init__(self, cities_filename):
        """
        Instantiate a StabilitasFilter object with cities specified in the
        passed file.
        """
        self.cities_df = self._load_cities(cities_filename)

    def _load_cities(self, cities_filename):
        """
        Load cities data from the given file.

        Input: filename
        Output: Self with stored city data.
        """
        pass

    def fit(self, filename):
        """
        Fit model to reports included in file. This method loads the data,
        preprocesses it, builds time series, and identifies anomalies.

        Input: filename
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
        print "Skeleton working"

if __name__ == '__main__':
    main()
