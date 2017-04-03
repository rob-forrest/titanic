""" Writing my first randomforest code.
Author : AstroDave
Date : 23rd September 2012
Revised: 15 April 2014
please see packages.python.org/milk/randomforests.html for more

""" 
import pandas as pd
import numpy as np
import csv as csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor




def extract_title(name):
    title_dict = {'Mr': 6, 'Mrs': 1, 'Miss': 2, 'Master': 3, 'Dr': 4, 'Rev': 5}
    for title in title_dict:
        if title in name:
            return title_dict[title]

    return 0




# Data cleanup
# TRAIN DATA
train_df = pd.read_csv('train.csv', header=0)        # Load the train file into a dataframe

# I need to convert all strings to integer classifiers.
# I need to fill in the missing values of the data and make it complete.

# female = 0, Male = 1
train_df['Gender'] = train_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# Embarked from 'C', 'Q', 'S'
# Note this is not ideal: in translating categories to numbers, Port "2" is not 2 times greater than Port "1", etc.

# All missing Embarked -> just make them embark from most common place
if len(train_df.Embarked[ train_df.Embarked.isnull() ]) > 0:
    train_df.Embarked[ train_df.Embarked.isnull() ] = train_df.Embarked.dropna().mode().values

Ports = list(enumerate(np.unique(train_df['Embarked'])))    # determine all values of Embarked,
Ports_dict = { name : i for i, name in Ports }              # set up a dictionary in the form  Ports : index
train_df.Embarked = train_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)     # Convert all Embark strings to int


#-----------------------------------------------------------------
# # All the ages with no data -> make the median of all Ages
# median_age = train_df['Age'].dropna().median()
# if len(train_df.Age[ train_df.Age.isnull() ]) > 0:
#     train_df.loc[ (train_df.Age.isnull()), 'Age'] = median_age

# Train a regression decision tree to predict age
# 1. prepare the features
train_df['Title'] = train_df['Name'].map(extract_title) # create new feature for passenger title



# 2. train the tree

# Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
train_df = train_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1)

# drop the samples without age
train_df_nona = train_df.dropna(axis=0)

# train a model for age
train_y = train_df_nona['Age'].values
train_X = train_df_nona.drop(['Age', 'Survived'], axis=1).values

dtr = DecisionTreeRegressor()
dtr.fit(train_X, train_y)


# 3. use tree to fill in blank values
train_df.loc[ (train_df.Age.isnull()), 'Age'] = dtr.predict(train_df[train_df.Age.isnull()].drop(['Age', 'Survived'], axis=1).values)



#-----------------------------------------------------------------






# TEST DATA
test_df = pd.read_csv('test.csv', header=0)        # Load the test file into a dataframe

# I need to do the same with the test data now, so that the columns are the same as the training data
# I need to convert all strings to integer classifiers:
# female = 0, Male = 1
test_df['Gender'] = test_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# Embarked from 'C', 'Q', 'S'
# All missing Embarked -> just make them embark from most common place
if len(test_df.Embarked[ test_df.Embarked.isnull() ]) > 0:
    test_df.Embarked[ test_df.Embarked.isnull() ] = test_df.Embarked.dropna().mode().values # try changing to train_df
# Again convert all Embarked strings to int
test_df.Embarked = test_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)




# All the missing Fares -> assume median of their respective class
if len(test_df.Fare[ test_df.Fare.isnull() ]) > 0:
    median_fare = np.zeros(3)
    for f in range(0,3):                                              # loop 0 to 2
        median_fare[f] = test_df[ test_df.Pclass == f+1 ]['Fare'].dropna().median()
    for f in range(0,3):                                              # loop 0 to 2
        test_df.loc[ (test_df.Fare.isnull()) & (test_df.Pclass == f+1 ), 'Fare'] = median_fare[f]

test_df['Title'] = test_df['Name'].map(extract_title)

# Collect the test data's PassengerIds before dropping it
ids = test_df['PassengerId'].values
# Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
test_df = test_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1)

# All the ages with no data -> make the median of all Ages
median_age = test_df['Age'].dropna().median() # try with train_df
if len(test_df.Age[ test_df.Age.isnull() ]) > 0:
    test_df.loc[ (test_df.Age.isnull()), 'Age'] = median_age
#test_df.loc[(test_df.Age.isnull()), 'Age'] = dtr.predict(test_df[test_df.Age.isnull()].drop(['Age'], axis=1).values)



# The data is now ready to go. So lets fit to the train, then predict to the test!
# Convert back to a numpy array
train_data = train_df.values
test_data = test_df.values

param_grid = [
  {'n_estimators': [10, 50, 100, 1000, 10000], 'criterion': ['gini', 'entropy'], 'max_features': ['auto', 'sqrt', 'log2', None]}
 ]
# param_grid = [
#   {'n_estimators': [5, 10, 100, 1000]}
#  ]

X_train = train_data[0::,1::] 
y_train = train_data[0::,0]
# clf = GridSearchCV(RandomForestClassifier(), param_grid) #, cv=5, scoring='precision_macro')
# clf.fit(X_train, y_train)
# print clf.best_params_
# print clf.cv_results_['mean_test_score']
# print clf.cv_results_['params']


forest = RandomForestClassifier(n_estimators=1000, max_features=None, criterion='entropy')
forest.fit(X_train, y_train)
print list(train_df.columns.values)
print forest.feature_importances_


# print 'Training...'
# forest = RandomForestClassifier(n_estimators=100)
# forest = forest.fit( train_data[0::,1::], train_data[0::,0] )

print 'Predicting...'
output = forest.predict(test_data).astype(int)

# NEXT STEPS (3/20/17): build a predictive model for age and fare to impute missing data
# NEXT STEPS (3/27/17): use decision tree to predict ages in training set and testing set,
    # then see how overall accuracy is affected
#NEXT STEPS (4/3/17): Does interpolating or using title for age do better than our DT?
    # Why did our DT for predicting missing age values not improve anything?


predictions_file = open("myfirstforest.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print 'Done.'








