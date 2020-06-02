import numpy as np

def get_distance(x1, x2):
    sum = 0
    if type(x1) == np.float64:
        sum += (x1 - x2) ** 2
    else:
        for i in range(len(x1)):
            sum += (x1[i] - x2[i]) ** 2
    return np.sqrt(sum)


def kmeans(X, k, max_iters, centers):
    centroids = X[np.random.choice(range(len(X)), k, replace=False)]
    if centers == 'random':
        return np.array(centroids), None
    converged = False
    current_iter = 0
    stds = np.zeros(k)

    while (not converged) and (current_iter < max_iters):
        cluster_list = [[] for _ in range(len(centroids))]
        for x in X:  # Go through each data point
            distances_list = []
            for c in centroids:
                distances_list.append(get_distance(c, x))
            cluster_list[int(np.argmin(distances_list))].append(x)

        # cluster_list = list((filter(None, cluster_list)))
        prev_centroids = centroids.copy()
        centroids = []

        for j in range(len(cluster_list)):
            if len(cluster_list[j]) != 0:
                centroids.append(np.mean(cluster_list[j], axis=0))
            else:
                centroids.append(prev_centroids[j])

        if np.array_equal(prev_centroids, centroids):
            converged = True
        current_iter += 1


    clustersWithNoPoints = []
    for i in range(k):
        if len(cluster_list[i]) < 2:
            # keep track of clusters with no points or 1 point
            clustersWithNoPoints.append(i)
            continue
        else:
            stds[i] = np.std(cluster_list[i])

    # if there are clusters with 0 or 1 points, take the mean std of the other clusters
    if len(clustersWithNoPoints) > 0:
        pointsToAverage = []
        for i in range(k):
            if i not in clustersWithNoPoints:
                pointsToAverage.append(cluster_list[i])
        pointsToAverage = np.concatenate(pointsToAverage).ravel()
        stds[clustersWithNoPoints] = np.mean(np.std(pointsToAverage))

    return np.array(centroids), stds


def rbf(x, c, s):
    return np.exp(-(get_distance(x, c)) ** 2 / (2 * s ** 2))


def betaBackprop(x ,c ,s):
    return -(get_distance(x, c)) ** 2 / (2 * s ** 4)


def centerBackprop(x , c, s):
    dis = (get_distance(x, c))
    if dis == 0:
        dis = np.exp(-51)
    return (s**2)*dis


class RBFNet(object):
    """Implementation of a Radial Basis Function Network"""
    def __init__(self, k=2, lr=0.01, momentum=0, epochs=100, rbf=rbf, inferStds='each', centers_init='kmeans', img=[], verifyX=[], verifyY=[]):
        self.k = k
        self.lr = lr
        self.momentum = momentum
        self.epochs = epochs
        self.rbf = rbf
        self.inferStds = inferStds
        self.centers_init = centers_init
        self.img = img
        self.images = []
        self.verifyX = verifyX
        self.verifyY = verifyY
        self.d_w_tmp = 0
        self.d_b_tmp = 0

        self.w = np.random.randn(k)
        self.b = np.random.randn(1)

    def fit(self, X, y, update='weight'):
        self.X = X
        self.genCentersAndStds(X)

        self.epochErrors = []
        self.selfVerify = []
        self.testVerify = []
        # training
        for epoch in range(self.epochs):
            sampleError = []
            for i in range(X.shape[0]):
                # forward pass
                a = np.array([self.rbf(X[i], c, s) for c, s, in zip(self.centers, self.stds)])
                F = a.T.dot(self.w) + self.b

                loss = (y[i] - F).flatten() ** 2
                sampleError.append(loss)

                # backward pass
                error = -(y[i] - F).flatten()

                # print(self.centers)

                delta_w = self.lr * a * error + (self.momentum * self.d_w_tmp)
                delta_b = self.lr * error + (self.momentum * self.d_b_tmp)
                if update == 'all':
                    delta_centers = (0.001 * a * error * self.w)/np.array([centerBackprop(X[i], c, s) for c, s, in zip(self.centers, self.stds)])
                    delta_stds = 0.01 * a * error * self.w * np.array([betaBackprop(X[i], c, s) for c, s, in zip(self.centers, self.stds)])
                    self.stds = self.stds - delta_stds
                    self.centers = self.centers - delta_centers

                # online update
                self.w = self.w - delta_w
                self.b = self.b - delta_b


                self.d_w_tmp = delta_w
                self.d_b_tmp = delta_b
            # print(self.epochErrors)
            if len(self.img) > 0 and epoch in self.img:
                self.draw()
            if len(self.verifyX) > 0:
                self.selfVerify.append(self.verifyPercentage(X,y))
                self.testVerify.append(self.verifyPercentage(self.verifyX, self.verifyY))

            self.epochErrors.append(np.mean(sampleError))

    def predict(self, X):
        y_pred = []
        for i in range(X.shape[0]):
            a = np.array([self.rbf(X[i], c, s) for c, s, in zip(self.centers, self.stds)])
            F = a.T.dot(self.w) + self.b
            y_pred.append(F)
        return np.array(y_pred)

    def errorForData(self, X , Y):
        output = self.predict(X)
        return np.mean(np.square(Y-output.flatten()))

    def reset(self):
        self.w = np.random.randn(self.k)
        self.b = np.random.randn(1)
        self.genCentersAndStds(self.X)

    def genCentersAndStds(self, X):
        if self.inferStds == 'each':
            # compute stds from data
            self.centers, self.stds = kmeans(X, self.k, 100, self.centers_init)
        elif self.inferStds in ['same', 'sameBig', 'sameSmall']:
            # use a fixed std
            self.centers, _ = kmeans(X, self.k, 100, self.centers_init)
            if self.k > 1:
                dMax = max([get_distance(c1, c2) for c1 in self.centers for c2 in self.centers])
                beta = dMax / np.sqrt(2 * self.k)
            else:
                beta = 1
            if self.inferStds == 'same':
                self.stds = np.repeat(beta, self.k)
            elif self.inferStds =='sameBig':
                self.stds = np.repeat(beta * 2, self.k)
            else:
                self.stds = np.repeat(beta / 4, self.k)
        elif self.inferStds == 'random':
            self.centers, _ = kmeans(X, self.k, 100, self.centers_init)
            self.stds = np.random.rand(self.k)
            print(self.stds)


    def draw(self):
        X_pred = np.arange(-4, 4, 0.1)
        y_pred = self.predict(X_pred)
        self.images.append(y_pred)

    def verifyPercentage(self, X, Y):
        return (np.sum(np.round(self.predict(X)).flatten() == Y)/Y.shape[0]) * 100