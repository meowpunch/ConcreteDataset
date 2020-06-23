import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils.logger import init_logger


def draw_hist(s, h_type: str = "dist", name: str = None):
    plt.figure()
    h_method = {
        "dist": sns.distplot,
        "count": sns.countplot,
    }
    try:
        method = h_method[h_type]
    except KeyError:
        # TODO: handle exception
        init_logger().critical("histogram type '{h_type}' is not supported".format(h_type=h_type))
        sys.exit()

    if isinstance(s, pd.Series):
        plt.title('{name} histogram'.format(name=s.name))
        method(s)
        # plt.show()
    else:
        # for jupyter notebook
        plt.title('{name} histogram'.format(name=name))
        return list(map(lambda series: method(series), s))
