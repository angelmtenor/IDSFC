import pandas as pd
from ggplot import *


def plot_weather_data(turnstile_weather):
    """
    You are passed in a dataframe called turnstile_weather. 
    Use turnstile_weather along with ggplot to make a data visualization
    focused on the MTA and weather data we used in assignment #3.  
    You should feel free to implement something that we discussed in class 
    (e.g., scatterplots, line plots, or histograms) or attempt to implement
    something more advanced if you'd like.  

    Here are some suggestions for things to investigate and illustrate:
     * Ridership by time of day or day of week
     * How ridership varies based on Subway station (UNIT)
     * Which stations have more exits or entries at different times of day
       (You can use UNIT as a proxy for subway station.)

    If you'd like to learn more about ggplot and its capabilities, take
    a look at the documentation at:
    https://pypi.python.org/pypi/ggplot/

    You can check out:
    https://s3.amazonaws.com/content.udacity-data.com/courses/ud359/turnstile_data_master_with_weather.csv

    To see all the columns and data points included in the turnstile_weather 
    dataframe. 

    However, due to the limitation of our Amazon EC2 server, we are giving you a random
    subset, about 1/3 of the actual data in the turnstile_weather dataframe.
    """

    turnstile_weather['weekday'] = pd.to_datetime(turnstile_weather['DATEn']).dt.dayofweek
    print(turnstile_weather.head(10))

    hour_df = turnstile_weather[['Hour', 'ENTRIESn_hourly']].groupby('Hour', as_index=False).mean()
    week_df = turnstile_weather[['weekday', 'ENTRIESn_hourly']].groupby('weekday', as_index=False).mean()

    unit_df = turnstile_weather[['UNIT', 'ENTRIESn_hourly']].groupby('UNIT', as_index=False).mean()
    # unit_df = unit_df.nlargest(100, 'ENTRIESn_hourly')

    hour_unit_df = turnstile_weather[['Hour', 'UNIT', 'ENTRIESn_hourly']].groupby(['Hour', 'UNIT'],
                                                                                  as_index=False).mean()

    max_hour_unit = hour_unit_df[
        hour_unit_df.groupby(['Hour'])['ENTRIESn_hourly'].transform(max) == hour_unit_df['ENTRIESn_hourly']]

    print(max_hour_unit)

    plot = []

    plot.append(ggplot(hour_df, aes(x='Hour', weight='ENTRIESn_hourly')) +
                geom_bar(stat='identity') + ylab("Avg. Number of entries") + ggtitle('Mean entries by hour'))

    plot.append(ggplot(week_df, aes(x='weekday', weight='ENTRIESn_hourly')) +
                geom_bar(stat='identity') + ylab("Avg. Number of entries / hour") + ggtitle(
        'Mean hourly entries by weekday (0 = sunday)'))

    plot.append(ggplot(unit_df, aes(x='UNIT', weight='ENTRIESn_hourly')) +
                geom_bar() + ylab("Avg. Number of entries / hour") + ggtitle('Mean hourly entries by subway station'))

    plot.append(
        ggplot(max_hour_unit, aes(x='Hour', weight='ENTRIESn_hourly', fill='UNIT')) + geom_bar(stat='identity') +
        ylab("Avg. Number of entries / hour") + ggtitle('Stations with most hourly entries'))

    print(plot)


if __name__ == "__main__":
    filename = 'turnstile_data_master_with_weather.csv'
    df = pd.read_csv(filename)
    plot_weather_data(df)
