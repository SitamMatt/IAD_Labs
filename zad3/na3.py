import numpy as np
import matplotlib.pyplot as plt
from zad3 import RBFNet, rbf

approx_data_1 = np.loadtxt("approximation_train_1.txt")
approx_test = np.loadtxt("approximation_test.txt")

X = approx_data_1[:, 0]
y = approx_data_1[:, 1]
X_test = approx_test[:, 0]
y_test = approx_test[:, 1]

epochs = 100
X_pred = np.arange(-4, 4, 0.1)
beta_data = ['sameSmall', 'same', 'sameBig']
#Na 3 podpunkt 1
# for sample in beta_data:
#     plt.scatter(X, y, label='train', s=6, c='black')
#     plt.scatter(X_test, y_test, label='test', s=1, c='grey', alpha=.5)
#     for i in range(5):
#         rbfnet = RBFNet(lr=0.01, k=i*10+1, epochs=epochs, inferStds=sample, centers_init='random')
#         rbfnet.fit(X, y)
#         y_pred = rbfnet.predict(X_pred)
#         plt.plot(X_pred, y_pred, label='RBF '+str(i*10+1))
#     plt.legend()
#     plt.tight_layout()
#     plt.show()
# print(rbfnet.epochErrors[-1])

#Na 3 podpunkt 2
all_z = np.empty((0,len(X_pred)), float)
k = 11
for sample in beta_data:
    plt.scatter(X, y, label='train', s=6, c='black')
    plt.scatter(X_test, y_test, label='test', s=1, c='grey', alpha=.5)
    rbfnet = RBFNet(lr=0.1, k=k, epochs=epochs, inferStds=sample, centers_init='random')
    rbfnet.fit(X, y)
    y_pred = rbfnet.predict(X_pred)
    plt.plot(X_pred, y_pred, label='RBF '+str(11) + sample)
    for i in range(k):
        z = np.array([rbf(x, rbfnet.centers[i], rbfnet.stds[i]) * rbfnet.w[i] for x in X_pred])
        plt.plot(X_pred, z, c='red')
        all_z = np.append(all_z, [z], axis=0)
    plt.legend()
    plt.tight_layout()
    plt.show()

#test
# for sample in beta_data:
#     plt.scatter(X, y, label='train', s=6, c='black')
#     plt.scatter(X_test, y_test, label='test', s=1, c='grey', alpha=.5)
#     for i in range(3):
#         rbfnet = RBFNet(lr=0.0075, k=i*10+1, epochs=1000, inferStds=sample, centers_init='random')
#         rbfnet.fit(X, y, update='all')
#         y_pred = rbfnet.predict(X_pred)
#         plt.plot(X_pred, y_pred, label='RBF '+str(i*10+1))
#     plt.legend()
#     plt.tight_layout()
#     plt.show()
# print(rbfnet.epochErrors[-1])

#Na 3 podpunkt 3
# outFile = open('na3pkt3.txt', 'a+')
# for i in range(1, 42, 5):
#     print(i)
#     train_data_error = []
#     test_data_error = []
#     rbfnet = RBFNet(lr=0.01, k=i, epochs=epochs, inferStds='same', centers_init='random')
#     for _ in range(100):
#         print(_)
#         rbfnet.fit(X, y)
#         train_data_error.append(rbfnet.epochErrors[-1])
#         test_data_error.append(rbfnet.errorForData(X_test, y_test))
#         rbfnet.reset()
#     outFile.write(str(i) + ' avg_train_error= ' + str(round(np.mean(train_data_error),5)) + ' div_train_error= ' + str(round(np.std(train_data_error), 5))
#                   + ' avg_test_error= ' + str(round(np.mean(test_data_error), 5)) + ' div_test_error= ' + str(round(np.std(train_data_error), 5)) +'\n')
# outFile.close()

#Na 3 podpunkt 4
# img_iter = [0,2,10,25,50]
# rbfnet = RBFNet(lr=0.01, k=16, epochs=epochs, inferStds='same', centers_init='random', img=img_iter)
# rbfnet.fit(X, y)
#
# for i in range(len(img_iter)):
#     plt.plot(X_pred, rbfnet.images[i], label=str(img_iter[i])+' iter')
# plt.plot(X_pred, rbfnet.predict(X_pred), label=str(epochs)+' iter')
#
# plt.legend()
# plt.tight_layout()
# plt.show()