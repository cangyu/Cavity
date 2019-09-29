import numpy as np
from matplotlib import pyplot as plt

data = np.loadtxt('history.txt')

t = data[:, 0]
div = np.log(np.abs(data[:, 1]))

plt.plot(t, div, '-')
plt.xlabel('t/s')
plt.ylabel(r'$log(|\nabla \cdot \vec{U}|)$')
plt.show()
