import numpy as np
import matplotlib.pyplot as plt


def activationFunction(d,r):
    return np.exp(-d**2/r**2*2)

def calculateZ(x,c,r):
    return activationFunction(np.abs(x-c), r)

def calculateOutput():
    return weights[0]+np.sum(radials,axis=0)

k=4
centers = np.random.uniform(0, 10, k)
radiuses = np.random.uniform(0, 1, k)
weights = np.random.uniform(-4, 4, k+1)
x = np.arange(0,10,0.01)
radials = np.empty((0,len(x)), float)

for i in range(k):
    z = np.array([])
    for x_sample in x:
        z = np.append(z, calculateZ(x_sample, centers[i], radiuses[i])*weights[i+1])
    plt.plot(x, z, color='red')
    radials = np.append(radials, np.array([z]), axis=0)
plt.plot(x, calculateOutput(), color='black')
plt.show()
