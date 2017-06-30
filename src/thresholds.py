from stabilitasfilter import StabilitasFilter
import matplotlib.pyplot as plt
plt.style.use("ggplot")

def main():
    # Filepaths assume running this script from stabilitas-thresholds/ dir
    cities_filename = "data/cities300000.csv"
    filter_layer = StabilitasFilter(cities_filename, cleaned=True)

    data_filename = "data/reports_12DEC16-26DEC16.tsv"
    filter_layer.fit(
        data_filename,
        resample_size=3,
        window_size="1w",
        anomaly_threshold=1,
        precalculated=True
    )


    filter_layer.test()
    plt.scatter(
        filter_layer.cities_timeseries["Berlin"].index,
        filter_layer.cities_timeseries["Berlin"].values
    )
    plt.xlim("2016-12-12", "2016-12-27")
    plt.show()


if __name__ == '__main__':
    main()
