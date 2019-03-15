import numpy as np

x1 = np.arange(9.0).reshape((3, 3))
print('x1', x1)

x2 = np.arange(3.0)
print('x2', x2)

print('subtracted', np.subtract(x1, x2))

