import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


class StabilitasFinder(object):
    def __init__(self, date_lookup=None, city_lookup=None):
        """
        Inputs:
            Date lookup dictionary from StabilitasFilter
            City lookup dictionary from StabilitasFilter
        Outputs:
            Compelted Date and City lookup dictionaries.
            {date:
                [elevated_cities],
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
        self.date_lookup = date_lookup
        self.city_lookup = city_lookup

    def load_data(self, filename):
        df = pd.read_csv(filename)
        df = df.reset_index()
        df["start_ts"] = pd.to_datetime(df["start_ts"])
        self.flagged_df = df.sort_values("start_ts")
        # print self.flagged_df.info()

    def label_critical_reports(self, cutoff=30):
        self.flagged_df["critical"] = np.zeros(len(self.flagged_df))

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
        # print critical_df.groupby("city").count()["critical"]
        print sum(self.flagged_df["critical"])

    def preprocesses_data(self):
        """
        Tokenize, lemmatize (or stem), vecotrize
        """
        X = self.flagged_df["title"]
        y = self.flagged_df["critical"]
        X_train, X_test, self.y_train, self.y_test = train_test_split(X, y)

        self.vectorizer = TfidfVectorizer(analyzer="word", stop_words="english")
        self.X_train = self.vectorizer.fit_transform(X_train)
        self.X_test = self.vectorizer.transform(X_test)


    def fit(self):
        """
        Use SKLearn Multinomial Naive Bayes
        """
        self.preprocesses_data()
        self.model = MultinomialNB()
        self.model.fit(self.X_train, self.y_train)

    def predict_proba(self):
        """
        Predict probability that a given report is critical, from MultinomialNB
        model.
        """
        self.probas = [prob[1] for prob in self.model.predict_proba(self.X_test)]

        # print self.probas


    def predict(self, X=None, y=None, threshold=0.37):
        """
        Classify reports as critical or non-critical, based off predicted
        probability and threshold.

        From initial testing, a threshold of 0.37 yields a true positive rate of
        0.802 and a false positive rate of 0.213.
        """
        self.predicted = [1 if prob > threshold else 0 for prob in self.probas]

        self.confusion_matrix(self.y_test, self.predicted)
        return self.predicted

    def get_critical_cities(self):
        pass
