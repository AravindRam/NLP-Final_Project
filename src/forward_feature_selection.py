from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
import pandas as pd
import numpy as np
from pandas import Series
from numpy.random import randint
from sklearn.cross_validation import KFold

def function(data):
    #data = pd.read_csv('csvDump.csv')
    kf = KFold(len(data), n_folds=5)
    i=0
    result = Series()
    for train_index, test_index in kf:
        training = data.ix[train_index, : ]
        testing = data.ix[test_index, : ]

        training_label = training.score1
        training_data = training.drop('score1', 1)

        testing_label = testing.score1
        testing_data = testing.drop('score1', 1)
        result[i] = function()
        i = i + 1
    return np.mean(result) # This should be the kappa value

def forward_feature_selection(max_feature = 25):
    data = pd.read_csv('csvDump.csv')

    features = pd.DataFrame()
    features = pd.DataFrame(data['score1'])
    added_features = ['score1']  # Never take score1
    for i in range(max_feature):
        #print('the value of i is ', i)
        kappa = -2 # Initialization
        best_feature = ''
        for item in data:
            if item not in added_features:
                value = function(pd.concat([features, data[item]], axis=1))
               # print('The value returned for item is ', value, item)

                if value > kappa:
                    if best_feature != '':
                        features = features.drop(best_feature, axis=1)
                        added_features.remove(best_feature)
                    features = pd.concat([features, data[item]], axis=1)
                    added_features.append(item)
                    kappa = value
                    best_feature = item

    added_features.remove('score1')
    print('Best features selected ', added_features, len(added_features))

forward_feature_selection()