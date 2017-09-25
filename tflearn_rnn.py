import tflearn
from tflearn.layers.core import input_data, fully_connected
from tflearn.data_utils import to_categorical
import numpy as np
import pandas
import tensorflow
tensorflow.reset_default_graph()


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
trainX = np.array(traindata[["c1", "c2", "c3", "c4", "c5"]].sort_values(by=1, ascending=False, axis=1))
trainX.sort()
trainX = trainX[:, ::-1]
ttrainY = np.array(traindata[["hand"]])#.flatten()
#np.ravel(trainY)
# convert to one-hot for target "hand"
trainY = np.zeros((len(ttrainY), 10))
for i in range(len(ttrainY)):
	trainY[i, ttrainY[i]] = 1


# same for test data
testX = np.array(testdata[["c1", "c2", "c3", "c4", "c5"]].sort_values(by=1, ascending=False, axis=1))
testX.sort()
testX = testX[:, ::-1]
ttestY = np.array(testdata[["hand"]])#.flatten()
#np.ravel(testY)
# convert to one-hot for target "hand"
testY = np.zeros((len(ttestY), 10))
for i in range(len(ttestY)):
	testY[i, ttestY[i]] = 1

# print(testX.shape)
print(trainX[1:5])
print(trainY[1:5])

# make the model
net = tflearn.input_data(shape=[None, 5])
net = tflearn.fully_connected(net, 32, activation='tanh', regularizer='L2', weight_decay=0.001)
net = tflearn.dropout(net, 0.8)
net = tflearn.fully_connected(net, 128, activation='tanh', regularizer='L2', weight_decay=0.001)
net = tflearn.dropout(net, 0.8)
net = tflearn.fully_connected(net, 128, activation='tanh')
net = tflearn.dropout(net, 0.8)
net = tflearn.fully_connected(net, 10, activation='softmax')
sgd = tflearn.SGD(learning_rate=0.1, lr_decay=0.96, decay_step=1000)
net = tflearn.regression(net, optimizer=sgd,
                         loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(trainX, trainY, n_epoch=10, validation_set=(testX[:1000], testY[:1000]),
	show_metric=True, run_id="dense_model_new")

model.save("tflearn-poker_rnn.model")

model.load('tflearn-poker_rnn.model')

print("Prediction:", model.predict(np.array([35, 34, 33, 32, 31]).reshape(-1,5)))
#print(test_X[15:19], test_y[15:19])
