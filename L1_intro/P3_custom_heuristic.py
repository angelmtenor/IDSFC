import numpy as np
import pandas


def custom_heuristic(file_path):
    """
    You are given a list of Titatic passengers and their associated
    information. More information about the data can be seen at the link below:
    http://www.kaggle.com/c/titanic-gettingStarted/data

    For this exercise, you need to write a custom heuristic that will take
    in some combination of the passenger's attributes and predict if the passenger
    survived the Titanic disaster.

    Can your custom heuristic beat 80% accuracy?

    The available attributes are:
    Pclass          Passenger Class
                    (1 = 1st; 2 = 2nd; 3 = 3rd)
    Name            Name
    Sex             Sex
    Age             Age
    SibSp           Number of Siblings/Spouses Aboard
    Parch           Number of Parents/Children Aboard
    Ticket          Ticket Number
    Fare            Passenger Fare
    Cabin           Cabin
    Embarked        Port of Embarkation
                    (C = Cherbourg; Q = Queenstown; S = Southampton)

    SPECIAL NOTES:
    Pclass is a proxy for socioeconomic status (SES)
    1st ~ Upper; 2nd ~ Middle; 3rd ~ Lower

    Age is in years; fractional if age less than one
    If the age is estimated, it is in the form xx.5

    With respect to the family relation variables (i.e. SibSp and Parch)
    some relations were ignored. The following are the definitions used
    for SibSp and Parch.

    Sibling:  brother, sister, stepbrother, or stepsister of passenger aboard Titanic
    Spouse:   husband or wife of passenger aboard Titanic (mistresses and fiancees ignored)
    Parent:   mother or father of passenger aboard Titanic
    Child:    son, daughter, stepson, or stepdaughter of passenger aboard Titanic

    Write your prediction back into the "predictions" dictionary. The
    key of the dictionary should be the passenger's id (which can be accessed
    via passenger["PassengerId"]) and the associating value should be 1 if the
    passenger survvied or 0 otherwise. 

    For example, if a passenger is predicted to have survived:
    passenger_id = passenger['PassengerId']
    predictions[passenger_id] = 1

    And if a passenger is predicted to have perished in the disaster:
    passenger_id = passenger['PassengerId']
    predictions[passenger_id] = 0

    You can also look at the Titantic data that you will be working with
    at the link below:
    https://s3.amazonaws.com/content.udacity-data.com/courses/ud359/titanic_data.csv
    """

    predictions = {}
    df = pandas.read_csv(file_path)
    for passenger_index, passenger in df.iterrows():
        passenger_id = passenger['PassengerId']

        # your code here


        predictions[passenger_id] = 0
        if passenger['Sex'] == 'female':
            predictions[passenger_id] = 1
        if passenger['Age'] < 4 and passenger['Parch'] > 0:  # 'Parch' > 0 is used for filtering missing ages
            predictions[passenger_id] = 1
        if passenger['Age'] < 19 and passenger['Pclass'] == 1 and passenger['Parch'] > 0:
            predictions[passenger_id] = 1
        if passenger['Age'] < 15 and passenger['Pclass'] == 2 and passenger['Parch'] > 0:
            predictions[passenger_id] = 1

    return predictions


f_path = 'titanic_data.csv'

prediction = custom_heuristic(f_path)

dframe = pandas.read_csv(f_path)
dframe['prediction'] = prediction.values()
accuracy = np.mean(list(dframe['prediction'] == dframe['Survived']))

print("Your heuristic is {:.2f}% accurate".format(accuracy * 100))
