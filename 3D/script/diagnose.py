import numpy as np
from matplotlib import pyplot as plt

data_path = "test/20200923-145643/ITER41.txt"

n_node = 14315
n_face = 153694
n_cell = 75470

raw_data = np.loadtxt(data_path, skiprows=2)

print(len(raw_data))
print(n_node + n_face + n_cell)

node_data = raw_data[0 : n_node]
print(len(node_data))
print(node_data.shape)

face_data = raw_data[n_node: n_node+n_face]
print(len(face_data))
print(face_data.shape)

cell_data = raw_data[n_node+n_face : n_node + n_face + n_cell]
print(len(cell_data))
print(cell_data.shape)

cell_p = cell_data[:, -2]
print(cell_p.shape)

print(cell_p[47720-1])

#plt.hist(cell_p)
#plt.show()

print(max(cell_p))
print(min(cell_p))

cell_p_sorted = np.sort(cell_p)
for i in range(20):
    print(i, cell_p_sorted[i])

for i in range(20):
    print(-i-1, cell_p_sorted[-i-1])
