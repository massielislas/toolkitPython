from __future__ import (absolute_import, division, print_function, unicode_literals)

from unittest import TestCase,TestLoader,TextTestRunner
from .matrix import Matrix
import numpy as np
import os
from toolkit import utils

class TestMatrix(TestCase):

    infinity = float("infinity")

    def setUp(self):

        # NOTE: for discrete attributes, at least one value must be a float in order for numpy array
        # functions to work properly.
        m = Matrix()
        m.attr_names = ['A', 'B', 'C']
        m.str_to_enum = [{}, {}, {'R': 0, 'G': 1, 'B': 2}]
        m.enum_to_str = [{}, {}, {0: 'R', 1: 'G', 2: 'B'}]
        m.data = np.array([[1.5, -6, 1.0],
                  [2.3, -8, 2],
                  [4.1, self.infinity, 2]])
        self.m = m

        m2 = Matrix()
        m2.attr_names = ['A', 'B', 'C', 'D', 'E']
        m2.str_to_enum = [{}, {}, {}, {}, {'R': 0, 'G': 1, 'B': 2}]
        m2.enum_to_str = [{}, {}, {}, {}, {0: 'R', 1: 'G', 2: 'B'}]
        m2.data = np.array([[0.0, 1.0, 2.0, 3.0, 0.0],
                   [0.1, 1.1, 2.1, 3.1, 1.0],
                   [0.2, 1.2, 2.2, 3.2, 1.0],
                   [0.3, 1.3, 2.3, 3.3, 2.0],
                   [0.4, 1.4, 2.4, 3.4, 2.0]])
        self.m2 = m2

    def test_init_from(self):
        m2 = Matrix(self.m, 1, 1, 2, 2)
        self.assertListEqual(m2.row(0).tolist(), [-8, 2])
        self.assertListEqual(m2.row(1).tolist(), [self.infinity, 2])

    def test_add(self):
        self.m.add(self.m2, 0, 2, 3)
        self.m.print()
        self.assertListEqual(self.m.row(3).tolist(), self.m2.row(0)[2:].tolist())
        self.m.add(self.m2, 3, 2, 3)
        self.m.print()
        self.assertListEqual(self.m.row(9).tolist(), self.m2.row(4)[2:].tolist())

    def test_set_size(self):
        m = Matrix()
        m.set_size(3, 4)
        self.assertEqual(m.rows, 3)
        self.assertEqual(m.cols, 4)

    def test_load_arff(self):
        t = Matrix()
        data_path = "./datasets/iris.arff"
        url = "http://axon.cs.byu.edu/data/uci_class/iris.arff"
        utils.save_arff(url, data_path)
        t.load_arff("datasets/iris.arff")
        self.assertListEqual(t.row(t.rows-1).tolist(), [5.9, 3.0, 5.1, 1.8, 2.0])

    def test_rows(self):
        self.assertEquals(self.m.rows, 3)

    def test_cols(self):
        self.assertEquals(self.m.cols, 3)

    def test_row(self):
        self.assertListEqual(self.m.row(1).tolist(), [2.3, -8, 2])

    def test_col(self):
        self.assertListEqual(self.m.col(1).tolist(), [-6, -8, self.infinity])

    def test_get(self):
        self.assertEquals(self.m.get(0, 2), 1)
        self.assertEquals(self.m.get(2, 0), 4.1)

    def test_set(self):
        self.m.set(2, 1, 2.5)
        self.assertEquals(self.m.get(2, 1), 2.5)

    def test_attr_name(self):
        name = self.m.attr_name(2)
        self.assertEquals(name, 'C')

    def test_set_attr_name(self):
        self.m.set_attr_name(2, 'Color')
        self.assertEquals(self.m.attr_name(2), 'Color')

    def test_attr_value(self):
        self.assertEquals(self.m.attr_value(2, 0), 'R')

    def test_value_count(self):
        self.assertEquals(self.m.value_count(1), 0)     # continuous
        self.assertEquals(self.m.value_count(2), 3)     # R, G, B

    def test_shuffle(self):
        self.m.shuffle()
        self.m.shuffle()
        self.m.shuffle()
        self.m.shuffle()
        self.m.shuffle()
        self.m.shuffle()
        pass

    def test_column_mean(self):
        self.assertAlmostEquals(self.m.column_mean(0), 2.6333, 4)
        self.assertAlmostEquals(self.m.column_mean(1), -7, 4)

    def test_column_min(self):
        self.assertEquals(self.m.column_min(0), 1.5)
        self.assertEquals(self.m.column_min(1), -8)

    def test_column_max(self):
        self.assertEquals(self.m.column_max(0), 4.1)
        self.assertEquals(self.m.column_max(1), -6)

    def test_most_common_value(self):
        self.assertEquals(self.m.most_common_value(0), 1.5)
        self.assertEquals(self.m.most_common_value(2), 2)


    def test_append_rows(self):
        test_matrix = Matrix(self.m)

        # Verify it works with other matrices
        test_matrix.append_rows(test_matrix)
        assert test_matrix.data.shape == (6,3)

        # Verify it works with numpy array
        test_matrix.append_rows(test_matrix.data)
        assert test_matrix.data.shape == (12,3)

        # Verify it works with 2D list
        test_matrix.append_rows(test_matrix.data)
        assert test_matrix.data.shape == (24,3)

        # Verify incompatible number of rows
        with self.assertRaises(Exception) as context:
            test_matrix.append_rows(self.m.data[:,:2])
        print(str(context.exception))
        self.assertTrue('Incompatible number of columns' in str(context.exception))


    def test_append_columns(self):
        test_matrix = Matrix(self.m)

        # Verify it works with other matrices
        test_matrix.append_columns(test_matrix)
        assert test_matrix.data.shape == (3,6)

        # Verify it works with numpy array
        test_matrix.append_columns(test_matrix.data)
        assert test_matrix.data.shape == (3,12)

        # Verify it works with 2D list
        test_matrix.append_columns(test_matrix.data)
        assert test_matrix.data.shape == (3,24)

        # Verify incompatible number of columns
        with self.assertRaises(Exception) as context:
            test_matrix.append_columns(self.m.data[:1,:])
        self.assertTrue('Incompatible number of rows' in str(context.exception))



suite = TestLoader().loadTestsFromTestCase(TestMatrix)
TextTestRunner(verbosity=2).run(suite)
