#!/usr/bin/env python

import matplotlib.pyplot as plt
import scipy.stats as st
import numpy as np
import math
import numpy as np
import pandas as pd


def get_bins_num(x):
    q25, q75 = np.percentile(x, [25, 75])
    if q25 == 0 and q75 == 0:
        bins = 20
    else:
        bin_width = 2 * (q75 - q25) * len(x) ** (-1 / 3)
        bins = round((max(x) - min(x)) / bin_width)
        print("Freedman–Diaconis number of bins:", bins)
    return bins


def get_statistics_list(x):
    print(
        f"mean: {round(np.mean(x),5)} median: {round(np.median(x),5)} variance: {round(np.var(x),5)} min: {round(np.min(x),5)} max: {round(np.max(x),5)}"
    )
    return round(np.median(x), 5), round(np.mean(x), 5)


def plot_hist(x, xlabel, median, mean, title=None, fontsize=15):
    fig = plt.figure(figsize=(12, 4))
    bins = get_bins_num(x)

    plt.subplot(1, 2, 1)
    plt.hist(x, alpha=0.8, density=False, bins=bins, label="counts")
    plt.axvline(median, color="r", linestyle="--", label="median")
    plt.axvline(mean, color="y", linestyle="--", label="mean")
    plt.ylabel("Counts", fontsize=fontsize)
    plt.legend(loc="best", fontsize=12)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    if title is not None:
        plt.title(str(title), fontsize=fontsize)
    else:
        plt.title("counts histogram", fontsize=fontsize)

    plt.subplot(1, 2, 2)
    plt.hist(x, alpha=0.8, density=True, bins=bins, label="frequencies")
    mn, mx = plt.xlim()
    plt.xlim(mn, mx)
    kde_xs = np.linspace(mn, mx, 300)
    kde = st.gaussian_kde(x)
    plt.plot(kde_xs, kde.pdf(kde_xs), label="PDF")
    plt.ylabel("Probability", fontsize=fontsize)
    plt.legend(loc="upper right", fontsize=12)
    plt.xlabel(xlabel, fontsize=fontsize)
    if title is not None:
        plt.title(str(title), fontsize=fontsize)
    else:
        plt.title("frequency histogram", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.show()


def plot_hist_from_list_pval(path, filename, xlabel, title=None, fontsize=15):
    my_file = open(path + "/" + filename, "r")
    data = my_file.read()
    data_into_list = data.split("\n")
    data_float = [float(x) for x in data_into_list if x.strip()]
    my_file.close()
    median, mean = get_statistics_list(data_float)
    plot_hist(data_float, xlabel, median, mean, title, fontsize=fontsize)
