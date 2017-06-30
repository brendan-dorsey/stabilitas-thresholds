from stabilitasfilter import StabilitasFilter

def main():
    cities_filename = "../data/cities300000.csv"
    filter_layer = StabilitasFilter(cities_filename)


    filter_layer.test()


if __name__ == '__main__':
    main()
