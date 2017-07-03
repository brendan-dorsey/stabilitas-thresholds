import numpy as np
import pandas as pd


class StabilitasFinder(object):
    def __init__(self):
        """

        Outputs:
            {date:
                [critical_cities]
                }
            {city:
                "location": (lat, long)
                "timeseries": city_timeseries
                "anomalies": city_anomaly_timeseries
                "elevated_reports": {date:
                    [elevated_reports]
                }
                "critical_reports": {date:
                    [critical_reports]
                }
            }
        """
        pass

    def load_data(self, filename):
        df = pd.read_csv(filename)
        df = df.reset_index()
        df["start_ts"] = pd.to_datetime(df["start_ts"])
        df["critical"] = np.zeros(len(df))
        self.flagged_df = df.sort_values("start_ts")
        # print self.flagged_df.info()

    def label_critical_reports(self, cutoff=30):

        for city in self.flagged_df["city"].unique():

            city_df = self.flagged_df[self.flagged_df["city"] == city]
            city_df = city_df.set_index("start_ts")

            next_day = pd.Timedelta(days=1)

            # print city_df.index
            # break
            for row in city_df.iterrows():
                index = row[1][0]

                report_time = row[0]
                stop_time = report_time + next_day

                future_reports = city_df[report_time:stop_time]
                if len(future_reports) >= cutoff:
                    self.flagged_df.loc[index, "critical"] = 1
        # print sum(self.flagged_df["critical"])
        critical_df = self.flagged_df[self.flagged_df["critical"] > 0]
        print critical_df["city"].unique()
        print sum(self.flagged_df["critical"])

    def preprocesses_data(self):
        """
        Tokenize, lemmatize (or stem), vecotrize
        """
        pass

    def tokenize(self):
        """
        Use tokenizer, either NLTK or SKLearn
        """
        pass

    def lemmatize(self):
        """
        Use NLTK WordNet Lemmatizer
        """
        pass

    def vectorize(self):
        """
        Use SKLearn TfIdf Vectorizer
        """
        pass

    def fit(self):
        """
        Use SKLearn Multinomial Naive Bayes
        """
        pass

    def predict_proba(self):
        """
        Predict probability that a given report is critical, from MultinomialNB
        model.
        """
        pass

    def predict(self, X, y=None, threshold=0.2):
        """
        Classify reports as critical or non-critical, based off predicted
        probability and threshold.
        """

    def get_critical_cities(self):
        pass
