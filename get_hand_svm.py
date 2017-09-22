from sklearn import svm
from sklearn import ensemble
from sklearn import neighbors
import pandas
import numpy as np
from time import time

'''
trainX = np.array(pandas.read_csv("poker-training.data"))[:,:10]
trainY = np.array(pandas.read_csv("poker-training.data"))[:,10:]
testX = np.array(pandas.read_csv("poker-testing.data"))[:1000,:10]
testY = np.array(pandas.read_csv("poker-testing.data"))[:1000,10:]
'''

cnames = ["suit1", "card1", "suit2", "card2", "suit3", "card3", "suit4", "card4",
                                                "suit5", "card5", "hand"]
traindata = pandas.read_csv("poker-training.data", names=cnames)
testdata = pandas.read_csv("poker-testing.data", names=cnames)

# tag each card so it has a unique number associated with it
traindata["c1"] = traindata["suit1"]*20 + traindata["card1"]
traindata["c2"] = traindata["suit2"]*20 + traindata["card2"]
traindata["c3"] = traindata["suit3"]*20 + traindata["card3"]
traindata["c4"] = traindata["suit4"]*20 + traindata["card4"]
traindata["c5"] = traindata["suit5"]*20 + traindata["card5"]

testdata["c1"] = testdata["suit1"]*20 + testdata["card1"]
testdata["c2"] = testdata["suit2"]*20 + testdata["card2"]
testdata["c3"] = testdata["suit3"]*20 + testdata["card3"]
testdata["c4"] = testdata["suit4"]*20 + testdata["card4"]
testdata["c5"] = testdata["suit5"]*20 + testdata["card5"]

# take new cols
testX = np.array(traindata[["c1", "c2", "c3", "c4", "c5"]].sort_values(by=1, ascending=False, axis=1))
testY = np.array(traindata[["hand"]])
trainX = np.array(testdata[["c1", "c2", "c3", "c4", "c5"]].sort_values(by=1, ascending=False, axis=1))
trainY = np.array(testdata[["hand"]])

print(testX, testY)

#classifier = svm.SVC(kernel="poly", C=100)
#classifier = ensemble.RandomForestClassifier(n_estimators=50)
#classifier = neighbors.KNeighborsClassifier(algorithm="auto", weights="distance", n_neighbors=15)
classifier = ensemble.AdaBoostClassifier(n_estimators=25, learning_rate=0.5)

t1 = time()
classifier.fit(trainX, trainY)
print "Training time: ", str(round(time()-t1, 3)), "s"

t2 = time()
prediction = classifier.predict(testX)
print "Prediction time: ", str(round(time()-t2, 3)), "s"

#print "10th: ", prediction[10]
#print "26th: ", prediction[26]
#print "50th: ", prediction[50]

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(prediction, testY)

print "Accuracy: ", str(accuracy)
print "Accuracy %: ", str(round(accuracy*100, 2))
