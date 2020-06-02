import numpy as np
import matplotlib.pyplot as plt

np.random.seed(123)
x, y = np.random.uniform(size=(100, 2)).T
z = np.exp(-x**2 - y**2)
levels = np.linspace(0, 1, 100)

cnt = plt.tricontourf(x, y, z, levels=levels, cmap="ocean")

# This is the fix for the white lines between contour levels
for c in cnt.collections:
    c.set_edgecolor("face")

plt.show()