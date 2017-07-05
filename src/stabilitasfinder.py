import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


class StabilitasFinder(object):
    def __init__(self):
        """
        Inputs:
            Date lookup dictionary from StabilitasFilter
            City lookup dictionary from StabilitasFilter
            Anomalous Reports DataFrame from StabiltiasFilter
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

        Workflow:
            Instantiate StabilitasFinder
            Load data from StabilitasFilter by calling load_data(**kwargs)
            If evaluating or training, label critical reports by calling
                label_critical_reports(cutoff)
            Fit model to training data by calling fit(mode)
                NB: mode options are "evaluate", "train", and "predict"
        """


    def load_data(self,
        source,
        date_lookup=None,
        city_lookup=None
    ):
        """
        Method to load data into Finder Layer. Data must be passed from
        StabilitasFilter as either a file or pandas DataFrame.
        """
        self.date_lookup = date_lookup
        self.city_lookup = city_lookup

        if isinstance(source, str):
            df = pd.read_csv(source)
            df = df.reset_index()
            df["start_ts"] = pd.to_datetime(df["start_ts"])
            self.flagged_df = df.sort_values("start_ts")
        else:
            df = source.reset_index()
            df["start_ts"] = pd.to_datetime(df["start_ts"])
            self.flagged_df = df.sort_values("start_ts")

    def label_critical_reports(self, cutoff=30):
        self.flagged_df["critical"] = np.zeros(len(self.flagged_df))

        for city in self.flagged_df["city"].unique():

            city_df = self.flagged_df[self.flagged_df["city"] == city]
            city_df = city_df.set_index("start_ts")

            next_day = pd.Timedelta(days=1)

            for row in city_df.iterrows():
                index = row[1][0]

                report_time = row[0]
                stop_time = report_time + next_day

                future_reports = city_df[report_time:stop_time]
                if len(future_reports) >= cutoff:
                    self.flagged_df.loc[index, "critical"] = 1

        critical_df = self.flagged_df[self.flagged_df["critical"] > 0]
        print critical_df.groupby("city").count().sort_values("critical", ascending=False)["critical"]
        print sum(self.flagged_df["critical"])

    def preprocesses_data(self, mode="evaluate"):
        """
        Tokenize, stem, and vecotrize report title text. Store "critical" label
        as target value for evaulation.

        kwargs:
        mode - str, "evaluate", "train" or "predict", default: "evaluate"
            Use "evaluate" for a train/test split when evalutating the model
            Use "train" to train the model on a complete dataset
            Use "predict" to use the model on a live dataset
        """
        X = self.flagged_df["title"]
        y = self.flagged_df["critical"]
        self.vectorizer = TfidfVectorizer(analyzer="word", stop_words="english")

        if mode == "evaluate":
            X_train, X_test, self.y_train, self.y_test = train_test_split(X, y)
            self.X_train = self.vectorizer.fit_transform(X_train)
            self.X_test = self.vectorizer.transform(X_test)
        elif mode == "train":
            self.X_train = self.vectorizer.fit_transform(X)
            self.y_train = y
        elif mode == "predict":
            self.X_train = None
            self.y_train = None
            self.X_test = self.vectotizer.transform(X)
            self.y_test = y


    def fit(self, mode="evaluate"):
        """
        Preprocess date using TF-IDF Vecotrization.  Fit an SKLearn Multinomial
        Naive Bayes classifier to the training data provided.

        kwargs:
        mode - str, "evaluate", "train" or "predict", default: "evaluate"
            Use "evaluate" for a train/test split when evalutating the model
            Use "train" to train the model on a complete dataset
            Use "predict" to use the model on a live dataset
        """
        self.preprocesses_data(mode)
        self.model = MultinomialNB()
        self.model.fit(self.X_train, self.y_train)

    def predict_proba(self, X=None):
        """
        Predict probability that a given report is critical, from MultinomialNB
        model.
        """
        if X == None:
            self.probas = [prob[1] for prob in
                self.model.predict_proba(self.X_test)
            ]
        else:
            self.probas = [prob[1] for prob in
                self.model.predict_proba(X)
        ]

        return self.probas


    def predict(self, X=None, y=None, threshold=0.37):
        """
        Classify reports as critical or non-critical, based off predicted
        probability and threshold.

        From initial testing, a threshold of 0.37 yields a true positive rate of
        0.802 and a false positive rate of 0.213.
        """
        self.predict_proba(X)

        self.predicted = [1 if prob > threshold else 0 for prob in self.probas]

        self.confusion_matrix = confusion_matrix(self.y_test, self.predicted)
        return self.predicted

    def _labeled_critical_cities_by_day(self):
        if (self.date_lookup == None) & (self.city_lookup == None):
            print "Needs lookup dicts from Filter Layer"
        else:
            for city in self.flagged_df["city"].unique:
                city_df = self.flagged_df[self.flagged_df["city"] == city]
                series = pd.Series(
                    city_df["critical"],
                    index=city_df["start_ts"]
                )
                daily_critical = series.resample("d").sum()
                for day in daily_critical.index:
                    if daily_critical[day] > 0:
                        if len(self.date_lookup[day.date()]) == 1:
                            self.date_lookup[day.date()].append([city])
                        else:
                            self.date_lookup[day.date()][1].append(city)

    def _predicted_critical_cities_by_day(self):
        if (self.date_lookup == None) & (self.city_lookup == None):
            print "Needs lookup dicts from Filter Layer"
        else:
            pass

    def get_critical_cities(self):
        pass
