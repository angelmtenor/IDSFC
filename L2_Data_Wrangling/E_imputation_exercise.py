import pandas


def imputation(filename):
    # Pandas dataframes have a method called 'fillna(value)', such that you can
    # pass in a single value to replace any NAs in a dataframe or series. You
    # can call it like this:
    #     dataframe['column'] = dataframe['column'].fillna(value)
    #
    # Using the numpy.mean function, which calculates the mean of a numpy
    # array, impute any missing values in our Lahman baseball
    # data sets 'weight' column by setting them equal to the average weight.
    #
    # You can access the 'weight' colum in the baseball data frame by
    # calling baseball['weight']

    baseball = pandas.read_csv(filename)

    # YOUR CODE GOES HERE
    mean_weight = baseball['weight'].mean()
    baseball['weight'] = baseball['weight'].fillna(mean_weight)
    return baseball


baseball_filename = 'baseballdatabank-2017.1/core/Master.csv'
print(imputation(baseball_filename)['weight'].head(30))
