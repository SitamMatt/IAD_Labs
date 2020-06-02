import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors

from zad3 import RBFNet, rbf

class_data = np.loadtxt("classification_train.txt")
class_test = np.loadtxt("classification_test.txt")

class_data_train = class_data[:, 0:-1]
class_data_very = class_data[:, -1]

test_data = class_test[:, 0:-1]
test_very = class_test[:, -1]


def getBin(numbers):
    array = []
    for i in numbers:
        array.append("{0:b}".format(i).zfill(4))
    return array


def getColums(array, columns):
    return array[:, [columns[j] == '1' for j in range(4)]]

def getHeatmap(train, grid_size):
    x_min = train[:, 0].min() - 0.2
    x_max = train[:, 0].max() + 0.2
    y_min = train[:, 1].min() - 0.2
    y_max = train[:, 1].max() + 0.2

    x = np.linspace(x_min, x_max, grid_size)
    y = np.linspace(y_min, y_max, grid_size)

    X, Y = np.meshgrid(x, y)

    return X, Y

array_all_features = getBin([1, 2, 4, 8, 3, 5, 6, 9, 10, 12, 7, 11, 13, 14, 15])
epochs = 100
#na 4 podpunkt 1
#
# for sample in array_all_features:
#     for i in range(1, 42, 10):
#         rbfNet = RBFNet(lr=0.01, k=i, epochs=epochs, inferStds='same', verifyX=getColums(test_data, sample), verifyY=test_very)
#         X = getColums(class_data_train, sample)
#         rbfNet.fit(X, class_data_very)
#         plt.plot(np.arange(len(rbfNet.selfVerify)), rbfNet.selfVerify, label='RBF ' + str(i) + ' self verify')
#         plt.plot(np.arange(len(rbfNet.testVerify)), rbfNet.testVerify, label='RBF ' + str(i) + ' test verify')
#     plt.title('Cechy '+sample)
#     plt.ylim(top=100,bottom=0)
#     if sample == '1111':
#         plt.legend()
#     plt.tight_layout()
#     plt.show()

#Na 4 podpunkt 2
# outFile = open('na4pkt2.txt', 'a+')
# for i in range(1, 42, 5):
#     print(i)
#     train_data_very = []
#     test_data_very = []
#     rbfnet = RBFNet(lr=0.01, k=i, epochs=epochs, inferStds='same')
#     for _ in range(100):
#         print(str(i)+' '+str(_))
#         rbfnet.fit(class_data_train, class_data_very)
#         train_data_very.append(rbfnet.verifyPercentage(class_data_train, class_data_very))
#         test_data_very.append(rbfnet.verifyPercentage(test_data, test_very))
#         rbfnet.reset()
#     outFile.write(str(i) + ' avg_train_very= ' + str(round(np.mean(train_data_very), 5)) + ' div_train_very= ' + str(round(np.std(train_data_very), 5))
#                          + ' avg_test_very= ' + str(round(np.mean(test_data_very), 5)) + ' div_test_very= ' + str(round(np.std(train_data_very), 5)) + '\n')
# outFile.close()

#Na 4 podpunkt 3
# array_2_features = array_all_features[4:10]
#
# grid_size = 125
#
# norm = matplotlib.colors.Normalize(1,3)
# colors = [[norm(1), 'C1'],
#           [norm(2), 'C2'],
#           [norm(3), 'C3']]
# cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors, N=125)
#
# for sample in array_2_features:
#     data_array = getColums(class_data_train, sample)
#     test_array = getColums(test_data, sample)
#     heatmap_x, heatmap_y = getHeatmap(data_array, grid_size)
#     rbfNet = RBFNet(lr=0.01, k=11, epochs=epochs, inferStds='same')
#     rbfNet.fit(data_array, class_data_very)
#
#     heatmap_pred = np.round(rbfNet.predict(np.array(list(zip(heatmap_x,heatmap_y)))))
#     # print(rbfNet.predict(np.array(list(zip(heatmap_x,heatmap_y)))))
#     # heatmap_pred = rbfNet.predict(np.array(list(zip(heatmap_x,heatmap_y))))
#
#     c = plt.pcolormesh(heatmap_x, heatmap_y, heatmap_pred, cmap=cmap)
#
#     for i in range(len(data_array)):
#         plt.scatter(data_array[i, 0], data_array[i, 1], c='C' + str(int(class_data_very[i])), edgecolors='white')
#
#     for i in range(len(test_array)):
#         plt.scatter(test_array[i, 0], test_array[i, 1], c='C' + str(int(test_very[i])), edgecolors='black')
#
#     plt.show()
