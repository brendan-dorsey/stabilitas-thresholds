import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
%matplotlib inline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from haversine import haversine
from itertools import izip
import csv


def main():
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

    target_columns = [
        "lat",
        "lon",
        "id",
        "title",
        "start_ts",
        "report_type",
    ]

    data_filepath = "../data/2016/all_2016.txt"

    # Load sample dataset
    df = pd.read_table(data_filepath, header=None, names=column_names)

    # Drop rows with NaNs (this is admittedly ugly and the final model will be much more precise)
    df.dropna(axis=0, how="any", inplace=True)

    # Drop columns we won't need
    df = df[target_columns]

    # Convert timestamps from Unix Epoch time to Date Time Groups
    df["start_ts"] = pd.to_datetime(df["start_ts"], unit="s", errors="ignore")

    # Build severity score columns
    # df["severity_score"] = df["severity"].map(severity_score)
    # df["severity_quadratic"] = df["severity"].map(severity_score_quadratic)
    # df["severity_log"] = df["severity"].map(severity_score_log)
    # df["severity_exp"] = df["severity"].map(severity_score_exp)

    # Trim reports from outside the specified date range
    start = pd.to_datetime("2016-01-01")
    end = pd.to_datetime("2017-01-01")
    df = df[df["start_ts"] > start]
    df = df[df["start_ts"] < end]

    cities_df = pd.read_csv("../data/cities300000.csv")

    df["lat_long"] = zip(df["lat"], df["lon"])
    cities_df["lat_long"] = zip(cities_df["latitude"], cities_df["longitude"])

    city_label_indices = []
    for report in df["lat_long"]:
        distances = [haversine(report, city) for city in cities_df["lat_long"]]
        city_label_indices.append(np.argmin(distances))
        
    city_labels = []
    for index in city_label_indices:
        city_labels.append(cities_df.ix[index, "name"])

    labels_df = pd.DataFrame(city_labels)
    labels_df.to_csv("../data/2016/2016_city_labels.csv", header=False, mode="w")



if __name__ == '__main__':
    main()
