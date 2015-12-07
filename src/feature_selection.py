from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from skll.metrics import kappa
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn import svm
import copy
import matplotlib.pyplot as plt
from sklearn.cross_validation import KFold

def kNN(train_X,train_Y,test_X,test_Y):
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(train_X, train_Y)
    predicted = neigh.predict(test_X)
    return kappa(test_Y, predicted, weights="quadratic", allow_off_by_one=False)

def naiveBayes(train_X,train_Y,test_X,test_Y):
    gnb = GaussianNB()
    predicted = gnb.fit(train_X, train_Y).predict(test_X)
    return kappa(test_Y, predicted, weights="quadratic", allow_off_by_one=False)

def decionTree(train_X,train_Y,test_X,test_Y):
    clf = tree.DecisionTreeClassifier()
    predicted = clf.fit(train_X, train_Y).predict(test_X)
    return kappa(test_Y, predicted, weights="quadratic", allow_off_by_one=False)

def svmFN(train_X,train_Y,test_X,test_Y):
    clf = svm.LinearSVC()
    clf.fit(train_X,train_Y)
    predicted = clf.predict(test_X)
    return kappa(test_Y, predicted, weights="quadratic", allow_off_by_one=False)

def gradientBoosting(train_X,train_Y,test_X,test_Y):

    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)
    clf.fit(train_X,train_Y)
    predicted = clf.predict(test_X)
    return kappa(test_Y, predicted, weights="quadratic", allow_off_by_one=False)


def kFold_cross_validation(data,argNo):
    kf = KFold(len(data), n_folds=5)
    result = []
    for train_index, test_index in kf:
        training = data.iloc[train_index,]
        testing = data.iloc[test_index,]

        training_label = training.score1
        training_data = training.drop('score1', 1)

        testing_label = testing.score1
        testing_data = testing.drop('score1', 1)
        if(argNo == 1):
            result.append(kNN(training_data,training_label,testing_data,testing_label))
        elif(argNo == 2):
            result.append(naiveBayes(training_data,training_label,testing_data,testing_label))
        elif(argNo == 3):
            result.append(decionTree(training_data,training_label,testing_data,testing_label))
        elif(argNo == 4):
            result.append(svmFN(training_data,training_label,testing_data,testing_label))
    return sum(result)/len(result)

def function(data):
    index = int(0.8 * len(data))
    training = data.iloc[0:index, :]
    testing = data.iloc[index:, :]

    training_label = training.score1
    training_data = training.drop('score1', 1)

    testing_label = testing.score1
    testing_data = testing.drop('score1', 1)

    #knnKappa = kNN(training_data,training_label,testing_data,testing_label)
    #nbKappa = naiveBayes(training_data,training_label,testing_data,testing_label)
    #treeKappa = decionTree(training_data,training_label,testing_data,testing_label)
    #svmKappa = svmFN(training_data,training_label,testing_data,testing_label)
    gbKappa = gradientBoosting(training_data,training_label,testing_data,testing_label)
    return gbKappa

def forward_feature_selection(max_feature = 25):
    data = pd.read_csv('csvDump1.csv')
    kappaValues =[]
    features = pd.DataFrame()
    features = pd.DataFrame(data['score1'])
    added_features = ['score1']  # Never take score1
    max_score = -100

    for i in range(max_feature):
        #print('the value of i is ', i)
        kappa = -2 # Initialization
        best_feature = ''
        for item in data:
            if item not in added_features:
                #value = kFold_cross_validation(pd.concat([features, data[item]], axis=1),2)
               # print('The value returned for item is ', value, item)
                value = function(pd.concat([features, data[item]], axis=1))
                if value > kappa:
                    if best_feature != '':
                        features = features.drop(best_feature, axis=1)
                        added_features.remove(best_feature)
                    features = pd.concat([features, data[item]], axis=1)
                    added_features.append(item)
                    kappa = value
                    kappaValues.append(kappa)
                    best_feature = item

                    if max_score < value:
                        max_score = value
                        max_feature_set = copy.deepcopy(added_features)

    added_features.remove('score1')
    print('Best features selected ', added_features, len(added_features))
    print("Kappa Values :", kappaValues)
    max_feature_set.remove('score1')
    print("Max index :", kappaValues.index(max(kappaValues)))
    print("Best feature set:", max_feature_set, max_score, len(max_feature_set))
    plt.plot(kappaValues,"c",label = 'Gradient Boosting')
    plt.legend(loc = 'lower right',shadow=True,fontsize='x-large')
    plt.annotate('Maximum Kappa Value',xy=(kappaValues.index(max(kappaValues)),max(kappaValues)),xytext=(kappaValues.index(max(kappaValues))-35, max(kappaValues)-0.3),arrowprops=dict(arrowstyle="->"))
    plt.xlabel('Number of features')
    plt.ylabel('Kappa Values')
    plt.title('Features vs Kappa Values')
    plt.savefig("GBM.png")
    plt.show()

forward_feature_selection()
