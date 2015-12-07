from sklearn.neighbors import KNeighborsClassifier
from skll.metrics import kappa
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn import svm
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

def kNN(train_X,train_Y,test_X,test_Y):
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(train_X, train_Y)
    predicted = neigh.predict(test_X)
    return predicted,kappa(test_Y, predicted, weights="quadratic", allow_off_by_one=False)

def naiveBayes(train_X,train_Y,test_X,test_Y):
    gnb = GaussianNB()
    predicted = gnb.fit(train_X, train_Y).predict(test_X)
    return predicted,kappa(test_Y, predicted, weights="quadratic", allow_off_by_one=False)

def decionTree(train_X,train_Y,test_X,test_Y):
    clf = tree.DecisionTreeClassifier()
    predicted = clf.fit(train_X, train_Y).predict(test_X)
    return predicted,kappa(test_Y, predicted, weights="quadratic", allow_off_by_one=False)

def svmFN(train_X,train_Y,test_X,test_Y):
    clf = svm.LinearSVC()
    clf.fit(train_X,train_Y)
    predicted = clf.predict(test_X)
    return predicted,kappa(test_Y, predicted, weights="quadratic", allow_off_by_one=False)

def gradientBoosting(train_X,train_Y,test_X,test_Y):

    clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)
    clf.fit(train_X,train_Y)
    predicted = clf.predict(test_X)
    return predicted,kappa(test_Y, predicted, weights="quadratic", allow_off_by_one=False)

data = pd.read_csv('csvDump10Test.csv')
index = int(0.8 * len(data))
training = data[data.score1!=4]
testing = data[data.score1==4]

training_label = training.score1
training_data = training.drop('score1', 1)

actual_test_labels = pd.read_csv('public_leaderboard_solution.csv')
actual_essay_set = actual_test_labels[actual_test_labels.essay_set==10]

testing_label = actual_essay_set.essay_score
testing_label = testing_label
testing_data = testing.drop('score1', 1)

predictedKnn,kappaValue = kNN(training_data,training_label,testing_data,testing_label)
print("K-NN: "+str(kappaValue))
predictedNB,kappaValue = naiveBayes(training_data,training_label,testing_data,testing_label)
print("Naive Bayes: "+str(kappaValue))
predictedTree,kappaValue = decionTree(training_data,training_label,testing_data,testing_label)
print("Decision Tree: "+str(kappaValue))
predictedSvm,kappaValue = svmFN(training_data,training_label,testing_data,testing_label)
print("SVM: "+str(kappaValue))
predictedGBM,kappaValue= gradientBoosting(training_data,training_label,testing_data,testing_label)
print("Gradient Boosting: "+str(kappaValue))

k = pd.concat([pd.DataFrame(predictedKnn), pd.DataFrame(predictedNB), pd.DataFrame(predictedTree), pd.DataFrame(predictedSvm), pd.DataFrame(predictedGBM)], axis=1)
finalPredicted = k.apply(lambda x: x.value_counts().idxmax(), axis=1)
print("Average Kappa Value: "+str(kappa(testing_label, finalPredicted, weights="quadratic", allow_off_by_one=False)))
























