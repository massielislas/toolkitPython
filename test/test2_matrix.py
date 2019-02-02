from __future__ import (absolute_import, division, print_function, unicode_literals)

from unittest import TestCase,TestLoader,TextTestRunner
from toolkit.arff import Arff
import numpy as np

try:
    infinity = float('inf')
except:
    infinity = 1e30000

m = Arff()
m.attr_names = ['A', 'B', 'C']
m.str_to_enum = [{}, {}, {'R': 0, 'G': 1, 'B': 2}]
m.enum_to_str = [{}, {}, {0: 'R', 1: 'G', 2: 'B'}]
m.data = np.array([[1.5, -6, 1.0],
          [2.3, -8, 2],
          [4.1, infinity, 2]])
m.value_counts = np.array([0.0, 0, 3])

m2 = Arff()
m2.attr_names = ['A', 'B', 'C', 'D', 'E']
m2.str_to_enum = [{}, {}, {}, {}, {'R': 0, 'G': 1, 'B': 2}]
m2.enum_to_str = [{}, {}, {}, {}, {0: 'R', 1: 'G', 2: 'B'}]
m2.data = np.array([[0.0, 1.0, 2.0, 3.0, 0.0],
           [0.1, 1.1, 2.1, 3.1, 1.0],
           [0.2, 1.2, 2.2, 3.2, 1.0],
           [0.3, 1.3, 2.3, 3.3, 2.0],
           [0.4, 1.4, 2.4, 3.4, 2.0]])
m2.value_counts = np.array([0.0, 0, 0, 0, 3])

features = Arff(m2, 0, 0, m2.rows, m2.cols - 1)
labels = Arff(m2, 0, m2.cols - 1, m2.rows, 1)

m2.print()
features.shuffle(labels)
features.print()
labels.print()

print("\n")
m2.normalize()
m2.print()

print("\n")
m2.add(m2, 0, 0, 5)
m2.print()
