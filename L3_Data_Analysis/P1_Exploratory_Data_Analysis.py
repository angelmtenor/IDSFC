import matplotlib.pyplot as plt
import pandas as pd


def entries_histogram(turnstile_weather):
    """
    Before we perform any analysis, it might be useful to take a
    look at the data we're hoping to analyze. More specifically, let's 
    examine the hourly entries in our NYC subway data and determine what
    distribution the data follows. This data is stored in a dataframe
    called turnstile_weather under the ['ENTRIESn_hourly'] column.

    Let's plot two histograms on the same axes to show hourly
    entries when raining vs. when not raining. Here's an example on how
    to plot histograms with pandas and matplotlib:
    turnstile_weather['column_to_graph'].hist()

    Your histogram may look similar to bar graph in the instructor notes below.

    You can read a bit about using matplotlib and pandas to plot histograms here:
    http://pandas.pydata.org/pandas-docs/stable/visualization.html#histograms

    You can see the information contained within the turnstile weather data here:
    https://s3.amazonaws.com/content.udacity-data.com/courses/ud359/turnstile_data_master_with_weather.csv
    """

    plt.figure()
    plt.axis([0, 6000, 0, 45000])
    # your code here to plot a histogram for hourly entries when it is not raining
    turnstile_weather[turnstile_weather['rain'] == 0]['ENTRIESn_hourly'].hist(bins=200, label='No rain')
    # your code here to plot a histogram for hourly entries when it is raining
    turnstile_weather[turnstile_weather['rain'] == 1]['ENTRIESn_hourly'].hist(bins=200, label='Rain')
    plt.legend()

    return plt


if __name__ == "__main__":
    df = pd.read_csv('MTA_Subway_turnstile/turnstile_data_master_with_weather.csv')
    entries_histogram(df).show()
