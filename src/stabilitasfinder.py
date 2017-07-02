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
        df["start_ts"] = pd.to_datetime(df["start_ts"])
        df = df.sort_values("start_ts")
        self.flagged_df = df.set_index("id")
        # print self.flagged_df.info()

    def label_critical_reports(self, cutoff=30):
        self.flagged_df["critical"] = np.zeros(len(self.flagged_df))
        for city in self.flagged_df["city"].unique():

            city_df = self.flagged_df[self.flagged_df["city"] == city]
            city_df = city_df.set_index("start_ts")
            # print city_df.head(10)

            next_day = pd.Timedelta(days=1)

            # print city_df.index
            # break
            for row in city_df.iterrows():
                index = row[1][2]

                report_time = row[0]
                stop_time = report_time + next_day

                future_reports = city_df[report_time:stop_time]
                if len(future_reports) >= cutoff:
                    self.flagged_df.set_value(index, "critical", 1)
        print sum(self.flagged_df["critical"])

    def preprocesses_data(self):
        pass

    def lemmatize(self):
        pass

    def vectorize(self):
        pass

    def fit(self):
        pass

    def predict_proba(self):
        pass

    def get_critical_cities(self):
        pass
