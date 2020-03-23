
# Imports
import numpy as np
import plotly.graph_objects as go
import statistics

#loading data for 3
# train_data = np.empty((0,4), float)
# # Reading training data from file
# with open('transformation.txt', "r") as inFile:
#     for x in range(4):
#         data = inFile.readline()
#         data = data.split()
#         train_data = np.append(train_data, np.array([data], dtype=float), axis=0)


# Activation function - Sigmoid function
def sigmoid(t):
    return 1 / (1 + np.exp(-t))


# Derivative of sigmoid
def sigmoid_derivative(p):
    return p * (1 - p)


# Class definition
class NeuralNetwork:
    def __init__(self, x, y, z, step, momentum, bias):
        self.input = x
        self.y = y
        self.weights1 = np.random.rand(self.input.shape[1], z)
        self.weights2 = np.random.rand(z, self.y.shape[1])
        self.output = np.zeros(y.shape)
        self.step = step
        self.momentum = momentum
        self.bias = bias
        self.d_w1_tmp = 0
        self.d_w2_tmp = 0
        self.layer1 = 0
        self.training_samples = self.input.shape[0]

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1)+self.bias)
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2)+self.bias)
        return self.layer2

    def backprop(self):
        d_weights2 = np.dot(self.layer1.T, 2 * (self.y - self.output) * sigmoid_derivative(self.output))
        d_weights1 = np.dot(self.input.T, np.dot(2 * (self.y - self.output) * sigmoid_derivative(self.output),
                                                 self.weights2.T) * sigmoid_derivative(self.layer1))

        self.weights1 += self.step * d_weights1 + self.momentum * self.d_w1_tmp
        self.weights2 += self.step * d_weights2 + self.momentum * self.d_w2_tmp

        self.d_w1_tmp = d_weights1
        self.d_w2_tmp = d_weights2

    def train(self, x, y):
        self.output = self.feedforward()
        self.backprop()

    def error(self):
        return (1/self.training_samples)*np.mean(np.square(self.y - self.feedforward()))

    def reset(self, z):
        self.weights1 = np.random.rand(self.input.shape[1], z)
        self.weights2 = np.random.rand(z, self.y.shape[1])

#first for 3
# np.random.seed(1)
# NN_3_1 = NeuralNetwork(train_data, train_data, 1, 0.1, 0, False)
# NN_3_2 = NeuralNetwork(train_data, train_data, 2, 0.1, 0, False)
# NN_3_3 = NeuralNetwork(train_data, train_data, 3, 0.1, 0, False)
# NN_3_1b = NeuralNetwork(train_data, train_data, 1, 0.1, 0, True)
# NN_3_2b = NeuralNetwork(train_data, train_data, 2, 0.1, 0, True)
# NN_3_3b = NeuralNetwork(train_data, train_data, 3, 0.1, 0, True)
# NN_3_1_error_data = []
# NN_3_2_error_data = []
# NN_3_3_error_data = []
# NN_3_1b_error_data = []
# NN_3_2b_error_data = []
# NN_3_3b_error_data = []
# index = []
#
# for i in range(100000):
#     NN_3_1_error_data.append(float(NN_3_1.error()))
#     NN_3_1.train(train_data, train_data)
#
#     NN_3_2_error_data.append(float(NN_3_2.error()))
#     NN_3_2.train(train_data, train_data)
#
#     NN_3_3_error_data.append(float(NN_3_3.error()))
#     NN_3_3.train(train_data, train_data)
#
#     NN_3_1b_error_data.append(float(NN_3_1b.error()))
#     NN_3_1b.train(train_data, train_data)
#
#     NN_3_2b_error_data.append(float(NN_3_2b.error()))
#     NN_3_2b.train(train_data, train_data)
#
#     NN_3_3b_error_data.append(float(NN_3_3b.error()))
#     NN_3_3b.train(train_data, train_data)
#
#     index.append(i)
#
# #graphs
# err = go.Figure()
# err.add_trace(go.Scatter(x=index, y=NN_3_1_error_data,
#                     line=dict(color='red'),
#                     name='1 neuron'))
# err.add_trace(go.Scatter(x=index, y=NN_3_2_error_data,
#                     line=dict(color='green'),
#                     name='2 neurons'))
# err.add_trace(go.Scatter(x=index, y=NN_3_3_error_data,
#                     line=dict(color='blue'),
#                     name='3 neurons'))
# err.update_layout(
#     title="Error at iterations without bias",
#     xaxis_title="Iteration",
#     yaxis_title="Error",
#     )
# err.show()
#
# err_b = go.Figure()
# err_b.add_trace(go.Scatter(x=index, y=NN_3_1b_error_data,
#                     line=dict(color='red'),
#                     name='1 neuron'))
# err_b.add_trace(go.Scatter(x=index, y=NN_3_2b_error_data,
#                     line=dict(color='green'),
#                     name='2 neurons'))
# err_b.add_trace(go.Scatter(x=index, y=NN_3_3b_error_data,
#                     line=dict(color='blue'),
#                     name='3 neurons'))
# err_b.update_layout(
#     title="Error at iterations with bias",
#     xaxis_title="Iteration",
#     yaxis_title="Error",
#     )
# err_b.show()

#second for 3
# NN_step_data = [3,3,2,1,0.1,1.5,2,4,0.3,2.5]
# NN_mom_data = [0,0.1,0.2,0.3,0.1,0,0.75,0.1,0.4,0.6]
# np.random.seed()
# outFile = open('avarage_learning.txt', 'a+')
# for i in range(10):
#     NN_3_curr_iter_data = []
#     step =  NN_step_data.pop()
#     momentum = NN_mom_data.pop()
#     NN_3 = NeuralNetwork(train_data, train_data, 2, step, momentum, True)
#     for j in range(100):
#         iterations = 0
#         while NN_3.error() > 0.01:
#             NN_3.train(train_data, train_data)
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
#     NN_3_2.train(train_data, train_data)
#
#     NN_3_2b.train(train_data, train_data)
#
# outFile2 = open('hidden_layer.txt', 'a+')
# outFile2.write(str(NN_3_2.layer1) + '\n')
# outFile2.write(str(NN_3_2b.layer1) + '\n')
# outFile2.close()
