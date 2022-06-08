#!/usr/bin/env python
import sys
import pandas as pd
import pickle as plk
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st


def plot_read_pos(path, geneID, bins, xmax):
    with open(path + "/" + geneID + "/" + "start1_pos", "rb") as fp:
        start1 = plk.load(fp)
    with open(path + "/" + geneID + "/" + "end1_pos", "rb") as fp:
        end1 = plk.load(fp)
    with open(path + "/" + geneID + "/" + "start2_pos", "rb") as fp:
        start2 = plk.load(fp)
    with open(path + "/" + geneID + "/" + "end2_pos", "rb") as fp:
        end2 = plk.load(fp)

    a = pd.DataFrame(start1, columns=["FWD start pos"]).describe()
    b = pd.DataFrame(end1, columns=["FWD end pos"]).describe()
    c = pd.DataFrame(start2, columns=["REV start pos"]).describe()
    d = pd.DataFrame(end2, columns=["REV end pos"]).describe()
    stats = pd.concat([a, b, c, d], axis=1)
    print(stats)

    # plotting
    fig = plt.figure(figsize=(15, 6))

    ###### FWD strand
    plt.subplot(1, 2, 1)
    x = start1
    y = end1
    plt.hist(x, density=True, alpha=0.8, bins=bins, label="FWD start pos")
    plt.hist(y, density=True, alpha=0.8, bins=bins, label="FWD end pos")
    mn, mx = plt.xlim()
    plt.xlim(mn, xmax)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=10)
    plt.legend(loc="upper right", fontsize=12)
    plt.ylabel("Probability", fontsize=15)
    plt.xlabel("start/end pos", fontsize=20)
    plt.title("FWD strand pos", fontsize=25)

    ###### REV strand
    plt.subplot(1, 2, 2)
    x = start2
    y = end2
    plt.hist(x, density=True, alpha=0.8, bins=bins, label="REV start pos")
    plt.hist(y, density=True, alpha=0.8, bins=bins, label="REV end pos")
    mn, mx = plt.xlim()
    plt.xlim(mn, xmax)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=10)
    plt.legend(loc="upper right", fontsize=12)
    plt.ylabel("Probability", fontsize=15)
    plt.xlabel("start/end pos", fontsize=20)
    plt.title("REV strand pos", fontsize=25)
    return pd.DataFrame(start1, columns=["FWD start pos"]).describe()
