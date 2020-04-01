
import numpy as np
import plotly.graph_objects as go
import statistics
from zadanie1_3 import NeuralNetwork

# loading data for 5
classification_train = np.empty((0,4), float)
classification_very = np.empty((0,3), int)
# Reading training data from file
with open('classification_train.txt', "r") as inFile:
    for x in range(90):
        data = inFile.readline()
        data = data.split()
        y = int(data.pop())
        if y == 1:
            classification_very = np.append(classification_very, np.array([[1, 0, 0]]), axis=0)
        if y == 2:
            classification_very = np.append(classification_very, np.array([[0, 1, 0]]), axis=0)
        if y == 3:
            classification_very = np.append(classification_very, np.array([[0, 0, 1]]), axis=0)
        classification_train = np.append(classification_train, np.array([data], dtype=float), axis=0)
# print(classification_train)
# print(classification_very)
# classification_train_1 = classification_train[:,0]
# classification_train_1 = np.reshape(classification_train_1, (-1,1))
#
# classification_train_2 = classification_train[:,1]
# classification_train_2 = np.reshape(classification_train_2, (-1,1))
#
# classification_train_3 = classification_train[:,2]
# classification_train_3 = np.reshape(classification_train_3, (-1,1))
#
# classification_train_4 = classification_train[:,3]
# classification_train_4 = np.reshape(classification_train_4, (-1,1))
#
# classification_train_12 = np.append(classification_train_1,classification_train_2,axis=1)
# classification_train_34 = np.append(classification_train_3,classification_train_4,axis=1)
# classification_train_1234 = np.append(classification_train_12,classification_train_34,axis=1)
# print(classification_train_1234)

classification_test_data = np.empty((0,4), float)
classification_test_very = np.empty((0,1), int)
with open('classification_test.txt', "r") as inFile:
    for x in range(93):
        data = inFile.readline()
        data = data.split()
        classification_test_very = np.append(classification_test_very, np.array([[int(data.pop())]]), axis=0)
        classification_test_data = np.append(classification_test_data, np.array([data], dtype=float), axis=0)
# print(classification_test_data)
# print(classification_test_very)

NN_5_4 = NeuralNetwork(classification_train,classification_very,8,0.001,0,True)
for i in range(10000):
    print(NN_5_4.error())
    NN_5_4.train()
print('procent testu: ' + str(NN_5_4.verify_percentage(classification_test_data,classification_test_very)))