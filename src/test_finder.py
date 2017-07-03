from stabilitasfinder import StabilitasFinder
import matplotlib.pyplot as plt
import pandas as pd
import time
from datetime import datetime
plt.style.use("ggplot")


def main():
    """
    Function to test implementation of Stabilitas Finder.
    """
    finder = StabilitasFinder()
    finder.load_data("data/flagged_reports.csv")
    finder.label_critical_reports()

    # print finder.flagged_df.info()
    # print finder.flagged_df.describe()
    # print finder.flagged_df.head()
    # print finder.flagged_df.tail()


if __name__ == '__main__':
    main()
