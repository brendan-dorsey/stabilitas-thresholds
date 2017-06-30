from stabilitasfilter import StabilitasFilter

def main():
    # Filepath assumes running this script from stabilitas-thresholds/ dir
    cities_filename = "data/cities300000.csv"
    filter_layer = StabilitasFilter(cities_filename, cleaned=True)


    filter_layer.test()


if __name__ == '__main__':
    main()
