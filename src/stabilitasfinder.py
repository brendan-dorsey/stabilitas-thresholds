import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_predict, KFold
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import time
from itertools import izip


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

    def load_data(
        self,
        source,
        date_lookup=None,
        city_lookup=None,
    ):
        """
        Method to load data into Finder Layer. Data must be passed from
        StabilitasFilter as either a filepath or pandas DataFrame. Lookups
        are dictionaries to be passed in from Finder Layer

        *** NEED TO REFACTOR ONCE LABELING FUNCTIONS ARE MIGRATED ***
        """
        start = time.time()
        print "Loading data for Finder..."
        self.date_lookup = date_lookup
        self.city_lookup = city_lookup

        # Check to see if data is coming from disk or from data engine
        if isinstance(source, str):
            df = pd.read_csv(source)
            df["start_ts"] = pd.to_datetime(df["start_ts"])
            self.flagged_df = df.sort_values("start_ts")
            try:
                self.flagged_df.drop("Unnamed: 0", axis=1, inplace=True)
            except:
                pass
        else:
            df = source
            df.loc[:, "start_ts"] = pd.to_datetime(df["start_ts"])
            self.flagged_df = df.sort_values("start_ts")

        # Reset multi-index to integer index
        self.flagged_df.reset_index(drop=True, inplace=True)
        self.flagged_df["index_copy"] = np.arange(len(self.flagged_df))

        # Build up dummy columns in separate dataframe
        # Dummies for severity
        self.dummies = pd.get_dummies(
            self.flagged_df["severity"],
            drop_first=True
        )

        # Dummies for city
        city_dum = pd.get_dummies(
            self.flagged_df["city"],
            drop_first=True
        )
        self.dummies = pd.merge(
            self.dummies,
            city_dum,
            left_index=True,
            right_index=True,
        )

        # Dummies for event type
        type_dum = pd.get_dummies(
            self.flagged_df["report_type"],
            drop_first=True
        )
        self.dummies = pd.merge(
            self.dummies,
            type_dum,
            left_index=True,
            right_index=True,
        )

        print "     Data loaded in {} seconds.".format(time.time()-start)

    def trim_dates(self, start_date, end_date=False):
        """
        Method to trim reports between [start, stop) datetimes.
        Can accept any format that can be read by pandas into
        datetime format.

        *** MAY NEED TO MOVE TO StabilitasFilter ***
        """
        start_date = pd.to_datetime(start_date)
        self.flagged_df = self.flagged_df[
            self.flagged_df["start_ts"] >= start_date
        ]

        if end_date:
            end_date = pd.to_datetime(end_date)
            self.flagged_df = self.flagged_df[
                self.flagged_df["start_ts"] < end_date
            ]

        # Trim dummies to match
        self.dummies = self.dummies[
            self.dummies.index.isin(self.flagged_df.index)
        ]

    def label_critical_reports(self, cutoff=30):
        """
        Method to label critical reports.

        *** NEEDS TO BE MOVED TO StabilitasFilter ***
        """
        start = time.time()
        print "Labeling critical reports..."
        self.flagged_df["critical"] = np.zeros(len(self.flagged_df))
        next_day = pd.Timedelta(days=1)

        total_cities = len(self.flagged_df["city"].unique())
        for i, city in enumerate(self.flagged_df["city"].unique()):
            if i % 10 == 0:
                current_time = time.time() - start
                total_time = (current_time * total_cities) / (i+1)
                print "     ~{0} seconds remaining for {1} cities".format(
                    int(round(total_time - current_time)),
                    total_cities - i
                )

            city_df = self.flagged_df[self.flagged_df["city"] == city]
            city_df = city_df.set_index("start_ts")

            for row in city_df.iterrows():
                index = row[1][-2]

                report_time = row[0]
                stop_time = report_time + next_day

                future_reports = city_df[report_time:stop_time]
                if len(future_reports) >= cutoff:
                    self.flagged_df.loc[index, "critical"] = 1

        # critical_df = self.flagged_df[self.flagged_df["critical"] > 0]
        # critical_df["title"].to_csv(
        #     "data/critical_titles.txt", sep=" ", mode="w"
        # )
        # print "Critical cities by number of critical reports:"
        # print critical_df.groupby("city").count().sort_values(
        #     "critical", ascending=False
        # )["critical"]
        # print ""
        # print "Total critical reports: ", sum(self.flagged_df["critical"])

        print "     Reports labeled in {} seconds.".format(time.time()-start)

    def preprocesses_data(self, mode="evaluate"):
        """
        Tokenize, stem, and vecotrize report title text. Store "critical" label
        as target value for evaulation.

        kwargs:
        mode - str, "evaluate", "train" or "predict", default: "evaluate"
            Use "evaluate" for a train/test split when evalutating the model
            Use "train" to train the model on a complete dataset
            Use "predict" to use the model on a live dataset

        *** NEED TO REFACTOR FOR BATCH PROCESSING FUNCTIONALITY ***
        """
        print "Preprocessing data..."
        X = self.flagged_df["title"]
        y = self.flagged_df["critical"]
        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            stop_words="english",
        )

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

    def fit(self):
        """
        Preprocess date using TF-IDF Vecotrization.  Fit an SKLearn Random
        Forest classifier to the training data provided.

        kwargs:
        mode - str, "evaluate", "train" or "predict", default: "evaluate"
            Use "evaluate" for a train/test split when evalutating the model
            Use "train" to train the model on a complete dataset
            Use "predict" to use the model on a live dataset
        """
        self.model = RandomForestClassifier(
            n_estimators=100,
            n_jobs=-1,
            max_depth=None,
            min_samples_split=10,
            min_samples_leaf=1,
            max_features=100
        )
        self.model.fit(self.X_train, self.y_train)

    def predict_proba(self, X=None):
        """
        Predict probability that a given report is critical, from MultinomialNB
        model.
        """
        if X is None:
            self.probas = [
                prob[1] for prob in self.model.predict_proba(self.X_test)
            ]
        else:
            self.probas = [
                prob[1] for prob in self.model.predict_proba(X)
            ]

        return self.probas

    def predict(self, X=None, y=None, threshold=0.14):
        """
        Classify reports as critical or non-critical, based off predicted
        probability and threshold.
        """
        self.predict_proba(X)

        self.predicted = [1 if prob > threshold else 0 for prob in self.probas]

        self.confusion_matrix = confusion_matrix(self.y_test, self.predicted)
        return self.predicted

    def cross_val_predict(self, thresholds=None, model_type="rfc"):
        """
        Cross validate and predict across full dataset.
        Default model is Gradient Boosting Classifier.
        Default threshold of 0.2164 is best for quadratic scoring.
        Default threshold of 0.2044 is best for volume scoring.
        """
        start = time.time()
        print "Generating cross-validated predictions..."
        X = self.flagged_df["title"].values
        X_meta = self.dummies.values
        y = self.flagged_df["critical"].values
        vectorizer = TfidfVectorizer(
            analyzer="word",
            stop_words="english",
            max_features=2500
        )
        kf = KFold(n_splits=5, shuffle=False)
        cv_probas = []
        cv_predicted = []
        models = {
            "nb": MultinomialNB(),
            "gbc": GradientBoostingClassifier(
                learning_rate=0.1,
                n_estimators=1000,
                max_depth=13,
                min_samples_split=3,
                min_samples_leaf=1,
                max_features=100,
                subsample=0.3
            ),
            "rfc": RandomForestClassifier(
                n_estimators=100,
                n_jobs=-1,
                max_depth=None,
                min_samples_split=10,
                min_samples_leaf=1,
                max_features=100
            ),
            "svm": SVC(
                kernel="linear",
                C=10.,
                probability=True,
            ),
            "logreg": LogisticRegression(
                solver="sag",
                n_jobs=-1
            )
        }

        if thresholds is None:
            if model_type == "gbc":
                thresholds = [0.2164]
            elif model_type == "rfc":
                thresholds = [0.125]

        for train_index, test_index in kf.split(y):
            # Split, train, test on titles
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            X_train = vectorizer.fit_transform(X_train)
            X_test = vectorizer.transform(X_test)
            model = models[model_type]
            model.fit(X_train, y_train)
            title_probas = [
                prob[1] for prob in model.predict_proba(X_test.toarray())
            ]

            # Split, train, test on metadata
            X_train_meta, X_test_meta = X_meta[train_index], X_meta[test_index]
            y_train_meta, y_test_meta = y[train_index], y[test_index]
            model_meta = RandomForestClassifier(
                n_estimators=100,
                n_jobs=-1,
                max_depth=None,
                min_samples_split=10,
                min_samples_leaf=1,
                max_features="sqrt"
            )
            model_meta.fit(X_train_meta, y_train_meta)
            meta_probas = [
                prob[1] for prob in model_meta.predict_proba(X_test_meta)
            ]
            # cv_probas.extend(meta_probas)

            # Average title and double weight meta probas for final output
            probas = [
                (title + 2*meta) / 3
                for title, meta in izip(title_probas, meta_probas)
            ]
            cv_probas.extend(probas)

        for threshold in thresholds:
            predictions = [1 if prob > threshold else 0 for prob in cv_probas]
            cv_predicted.append(predictions)

        if len(cv_predicted) == 1:
            self.flagged_df["predicted"] = cv_predicted[0]
            self.flagged_df["predicted_probas"] = cv_probas
            self.confusion_matrix = confusion_matrix(
                self.flagged_df["critical"].values,
                self.flagged_df["predicted"].values
            )
            self.feature_importances_ = model.feature_importances_
            self.vocabulary_ = vectorizer.vocabulary_

        print "     Predictions made in {} seconds.".format(time.time()-start)
        return cv_predicted

    def extract_critical_titles(self):
        """
        Method to save critical titles to disk for qualitative analysis.
        """
        critical_df = self.flagged_df[
            (self.flagged_df["critical"] == 1) |
            (self.flagged_df["predicted"] == 1)
        ]

        critical_df[
            ["title", "critical", "predicted"]
        ].to_csv("eda/critical_titles.csv", sep=",", mode="w")

    def _labeled_critical_cities_by_day(self):
        """
        Method to add critical cities to the date_lookup dictionary.
        """
        if (self.date_lookup is None) | (self.city_lookup is None):
            print "Needs lookup dicts from Filter Layer"
        else:
            print "Grouping labeled cities by day..."
            for city in self.flagged_df["city"].unique():
                city_df = self.flagged_df[self.flagged_df["city"] == city]
                series = pd.Series(
                    city_df["critical"].values,
                    index=city_df["start_ts"],
                    copy=True
                )
                daily_critical = series.resample("d").sum()
                for day in daily_critical.index:
                    key = str(day.date())
                    try:
                        self.date_lookup[key]
                    except KeyError:
                        self.date_lookup[key] = [[]]
                    if len(self.date_lookup[key]) == 1:
                        self.date_lookup[key].append([])
                    if daily_critical[day] > 0.0:
                        self.date_lookup[key][1].append(city)

    def _predicted_critical_cities_by_day(self):
        """
        Method to add predicted critical cities to the date_lookup dictionary.
        """
        if (self.date_lookup is None) | (self.city_lookup is None):
            print "Needs lookup dicts from Filter Layer"
        else:
            print "Grouping predicted cities by day..."
            for city in self.flagged_df["city"].unique():
                city_df = self.flagged_df[self.flagged_df["city"] == city]
                series = pd.Series(
                    city_df["predicted"].values,
                    index=city_df["start_ts"],
                    copy=True
                )
                daily_predicted = series.resample("d").sum()
                for day in daily_predicted.index:
                    key = str(day.date())
                    if len(self.date_lookup[key]) == 1:
                        self.date_lookup[key].append([], [])
                    elif len(self.date_lookup[key]) == 2:
                        self.date_lookup[key].append([])
                    if daily_predicted[day] > 0.0:
                        self.date_lookup[key][2].append(city)

    def _most_critical_report_per_city_per_day(self):
        """
        Method to add predicted probabilities and most critical report
        headlines to city lookup dictionary.
        """
        if (self.date_lookup is None) | (self.city_lookup is None):
            print "Needs lookup dicts from Filter Layer"
        else:
            print "Extracting most critical reports..."
            for city in self.city_lookup.keys():
                city_df = self.flagged_df[self.flagged_df["city"] == city]
                city_df.loc[:, "date"] = city_df["start_ts"].apply(
                    lambda x: x.date()
                )
                days = set(city_df["date"])

                for day in days:
                    day_reports_df = city_df[city_df["date"] == day]
                    key = str(day)
                    index = np.argmax(day_reports_df["predicted_probas"])
                    proba = day_reports_df.loc[index, "predicted_probas"]
                    title = day_reports_df.loc[index, "title"]
                    try:
                        self.city_lookup[city][key] = (proba, title)
                    except KeyError:
                        self.city_lookup[city][key] = (
                            0, "None predicted critical"
                        )
