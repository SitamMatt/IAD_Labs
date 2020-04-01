import numpy as np
import plotly.graph_objects as go
import statistics
from zadanie1_3 import NeuralNetwork

# loading data for 3
train_data = np.empty((0,4), float)
# Reading training data from file
with open('transformation.txt', "r") as inFile:
    for x in range(4):
        data = inFile.readline()
        data = data.split()
        train_data = np.append(train_data, np.array([data], dtype=float), axis=0)

#first for 3
np.random.seed(1)
NN_3_1 = NeuralNetwork(train_data, train_data, 1, 0.1, 0, False)
NN_3_2 = NeuralNetwork(train_data, train_data, 2, 0.1, 0, False)
NN_3_3 = NeuralNetwork(train_data, train_data, 3, 0.1, 0, False)
NN_3_1b = NeuralNetwork(train_data, train_data, 1, 0.1, 0, True)
NN_3_2b = NeuralNetwork(train_data, train_data, 2, 0.1, 0, True)
NN_3_3b = NeuralNetwork(train_data, train_data, 3, 0.1, 0, True)
NN_3_1_error_data = []
NN_3_2_error_data = []
NN_3_3_error_data = []
NN_3_1b_error_data = []
NN_3_2b_error_data = []
NN_3_3b_error_data = []
index = []

for i in range(10000):
    NN_3_1_error_data.append(float(NN_3_1.error()))
    NN_3_1.train()

    NN_3_2_error_data.append(float(NN_3_2.error()))
    NN_3_2.train()

    NN_3_3_error_data.append(float(NN_3_3.error()))
    NN_3_3.train()

    NN_3_1b_error_data.append(float(NN_3_1b.error()))
    NN_3_1b.train()

    NN_3_2b_error_data.append(float(NN_3_2b.error()))
    NN_3_2b.train()

    NN_3_3b_error_data.append(float(NN_3_3b.error()))
    NN_3_3b.train()

    index.append(i)

#graphs
err = go.Figure()
err.add_trace(go.Scatter(x=index, y=NN_3_1_error_data,
                    line=dict(color='red'),
                    name='1 neuron'))
err.add_trace(go.Scatter(x=index, y=NN_3_2_error_data,
                    line=dict(color='green'),
                    name='2 neurons'))
err.add_trace(go.Scatter(x=index, y=NN_3_3_error_data,
                    line=dict(color='blue'),
                    name='3 neurons'))
err.update_layout(
    title="Error at iterations without bias",
    xaxis_title="Iteration",
    yaxis_title="Error",
    )
err.show()

err_b = go.Figure()
err_b.add_trace(go.Scatter(x=index, y=NN_3_1b_error_data,
                    line=dict(color='red'),
                    name='1 neuron'))
err_b.add_trace(go.Scatter(x=index, y=NN_3_2b_error_data,
                    line=dict(color='green'),
                    name='2 neurons'))
err_b.add_trace(go.Scatter(x=index, y=NN_3_3b_error_data,
                    line=dict(color='blue'),
                    name='3 neurons'))
err_b.update_layout(
    title="Error at iterations with bias",
    xaxis_title="Iteration",
    yaxis_title="Error",
    )
err_b.show()

#second for 3
# NN_step_data = [3,3,2,1,0.1,1.5,2,4,0.3,2.5]
# NN_mom_data = [0,0.1,0.2,0.3,0.1,0,0.75,0.1,0.4,0.6]
# np.random.seed()
# outFile = open('avarage_learning.txt', 'a+')
# for i in range(10):
#     NN_3_curr_iter_data = []
#     step = NN_step_data.pop()
#     momentum = NN_mom_data.pop()
#     NN_3 = NeuralNetwork(train_data, train_data, 2, step, momentum, True)
#     for j in range(100):
#         iterations = 0
#         while NN_3.error() > 0.01:
#             NN_3.train()
#             iterations += 1
#         NN_3_curr_iter_data.append(iterations)
#         NN_3.reset(2)
#     print(round(np.mean(NN_3_curr_iter_data)))
#     print(round(statistics.pstdev(NN_3_curr_iter_data)))
#     outFile.write(str(step) + ' ' + str(momentum) + ' ' + str(round(np.mean(NN_3_curr_iter_data))) + ' ' + str(round(statistics.pstdev(NN_3_curr_iter_data))) + '\n')
# outFile.close()

# third for 3
# np.random.seed(1)
# NN_3_2 = NeuralNetwork(train_data, train_data, 2, 2, 0, False)
# NN_3_2b = NeuralNetwork(train_data, train_data, 2, 2, 0, True)
#
# for i in range(1000000):
#     NN_3_2.train()
#
#     NN_3_2b.train()
#
# outFile2 = open('hidden_layer.txt', 'a+')
# outFile2.write(str(NN_3_2.layer1) + '\n')
# outFile2.write(str(NN_3_2b.layer1) + '\n')
# outFile2.close()