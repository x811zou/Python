import matplotlib.pyplot as plt
import pandas as pd

def plot_hist_from_df(df,columnname,title,xlabel,ylabel='histogram',bins=30):
    plt.hist(df[str(columnname)], bins=bins)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def plot_bar_chart(x,y,label,title,xlabel,ylabel='Count'):
    # create bar chart
    plt.bar(x, y,label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

def plot_scatter(x,y,xlabel,ylabel,title):
    plt.scatter(x,y)
    # Add labels and title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    # Show plot
    plt.show()