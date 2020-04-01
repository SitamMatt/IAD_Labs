import numpy as np
import plotly.graph_objects as go
import statistics
from zadanie1_3 import NeuralNetwork

approximation_train_1 = np.empty((0,1), float)
approximation_very_1 = np.empty((0,1), float)
# Reading training data from file
with open('approximation_train_1.txt', "r") as inFile:
    for x in range(81):
        data = inFile.readline()
        data = data.split()
        approximation_very_1 = np.append(approximation_very_1, np.array([[data.pop()]], dtype=float), axis=0)
        approximation_train_1 = np.append(approximation_train_1, np.array([[data.pop()]], dtype=float), axis=0)
# print(approximation_train_1)
# print(approximation_very_1)


approximation_train_2 = np.empty((0,1), float)
approximation_very_2 = np.empty((0,1), float)
# Reading training data from file
with open('approximation_train_2.txt', "r") as inFile:
    for x in range(15):
        data = inFile.readline()
        data = data.split()
        approximation_very_2 = np.append(approximation_very_2, np.array([[data.pop()]], dtype=float), axis=0)
        approximation_train_2 = np.append(approximation_train_2, np.array([[data.pop()]], dtype=float), axis=0)
# print(approximation_train_2)
# print(approximation_very_2)

approximation_test_data = np.empty((0,1), float)
approximation_test_very = np.empty((0,1), float)
# Reading training data from file
with open('approximation_test.txt', "r") as inFile:
    for x in range(1000):
        data = inFile.readline()
        data = data.split()
        approximation_test_very = np.append(approximation_test_very, np.array([[data.pop()]], dtype=float), axis=0)
        approximation_test_data = np.append(approximation_test_data, np.array([[data.pop()]], dtype=float), axis=0)
# print(approximation_test_data)
# print(approximation_test_very)

def nonlin(x, deriv=False):
    if (deriv == True):
        return x * (1 - x)

    return 1 / (1 + np.exp(-x))

# randomly initialize our weights
syn0 = np.random.random((1, 13))
syn1 = np.random.random((13, 1))

for j in range(60000):

    # Feed forward through layers 0, 1, and 2
    l0 = approximation_test_data
    l1 = nonlin(np.dot(l0, syn0)+1)
    l2 = np.dot(l1, syn1)+1

    # how much did we miss the target value?
    l2_error = approximation_test_very - l2

    if (j % 10000) == 0:
        print("Error:" + str(np.mean(np.square(l2_error))))

    # in what direction is the target value?
    # were we really sure? if so, don't change too much.
    l2_delta = 2 * l2_error

    # how much did each l1 value contribute to the l2 error (according to the weights)?
    l1_error = (2*l2_delta).dot(syn1.T)

    # in what direction is the target l1?
    # were we really sure? if so, don't change too much.
    l1_delta = l1_error * nonlin(l1, deriv=True)

    syn1 += l1.T.dot(l2_delta*0.00005)
    syn0 += l0.T.dot(l1_delta*0.00005)

l0 = approximation_test_data
l1 = nonlin(np.dot(l0, syn0) + 1)
l2 = np.dot(l1, syn1) + 1
l2_error = approximation_test_very - l2
print("Error2:" + str(np.mean(np.abs(l2_error))))
# NN_4_1 = NeuralNetwork(approximation_train_1, approximation_very_1, 1, 0.00117, 0.001, True)
# NN_4_2 = NeuralNetwork(approximation_train_1, approximation_very_1, 5, 0.00117, 0.001, True)
# NN_4_3 = NeuralNetwork(approximation_train_1, approximation_very_1, 9, 0.00117, 0.001, True)
# NN_4_4 = NeuralNetwork(approximation_train_1, approximation_very_1, 13, 0.00117, 0.001, True)
# NN_4_5 = NeuralNetwork(approximation_train_1, approximation_very_1_sig, 20, 0.1, 0, True)
# for i in range(100000):
    # if i % 10000 == 0:
    #     print(NN_4.error())
    # NN_4_1.train()
    # NN_4_2.train()
    # NN_4_3.train()
    # NN_4_4.train()
    # NN_4_5.train()

# print('blad testu: ' + str(NN_4.verify(approximation_test_data,approximation_test_very)))
# y_1 = np.empty((0,1), float)
# y_2 = np.empty((0,1), float)
# y_3 = np.empty((0,1), float)
# y_4 = np.empty((0,1), float)
y_5 = np.empty((0,1), float)

# y_1 = np.append(y_1,NN_4_1.feed(approximation_test_data),axis=0)
# y_2 = np.append(y_2,NN_4_2.feed(approximation_test_data),axis=0)
# y_3 = np.append(y_3,NN_4_3.feed(approximation_test_data),axis=0)
# y_4 = np.append(y_4,NN_4_4.feed(approximation_test_data),axis=0)
y_5 = np.append(y_5,l2,axis=0)

err = go.Figure()
# err.add_trace(go.Scatter(x=approximation_test_data.flatten(), y=y_1.flatten(),
#                     mode='markers',
#                     marker=dict(color='Yellow'),
#                     name='1 Neuron'))
# err.add_trace(go.Scatter(x=approximation_test_data.flatten(), y=y_2.flatten(),
#                     mode='markers',
#                     marker=dict(color='Green'),
#                     name='5 Neurons'))
# err.add_trace(go.Scatter(x=approximation_test_data.flatten(), y=y_3.flatten(),
#                     mode='markers',
#                     marker=dict(color='Brown'),
#                     name='9 Neurons'))
# err.add_trace(go.Scatter(x=approximation_test_data.flatten(), y=y_4.flatten(),
#                     mode='markers',
#                     marker=dict(color='Red'),
#                     name='13 Neurons'))
err.add_trace(go.Scatter(x=approximation_test_data.flatten(), y=l2.flatten(),
                    mode='markers',
                    marker=dict(color='Blue'),
                    name='17 Neurons'))
err.add_trace(go.Scatter(x=approximation_test_data.flatten(), y=approximation_test_very.flatten(),
                    mode='markers',
                    marker=dict(color='Purple'),
                    name='Data1'))
err.update_layout(
    title="Approximation for data1",
    xaxis_title="X",
    yaxis_title="Y",
    )
err.show()