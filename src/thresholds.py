from stabilitasfilter import StabilitasFilter

def main():
    # Filepaths assume running this script from stabilitas-thresholds/ dir
    cities_filename = "data/cities300000.csv"
    filter_layer = StabilitasFilter(cities_filename, cleaned=True)

    data_filename = "data/reports_12DEC16-26DEC16.tsv"
    filter_layer.fit(
        data_filename,
        resample_size=3,
        window_size="1w",
        anomaly_threshold=1
    )


    filter_layer.test()


if __name__ == '__main__':
    main()
