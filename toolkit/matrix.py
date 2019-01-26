from __future__ import (absolute_import, division, print_function, unicode_literals)

import random
import numpy as np
from scipy import stats
import re

class Matrix:

    """
    Unlike the Matrix class defined in toolkit (C++) and toolkitJava,
    data is an array of columns rather than an array of rows.
    This makes computing statistics on columns more efficient.

    For discrete attributes, at least one value must be a float in
    order for numpy array functions to work properly. (The load_arff
    function ensures that all values are read as floats.)
    """
    def __init__(self, matrix=None, row_start=None, col_start=None, row_count=None, col_count=None, arff=None):
        """
        If matrix is provided, all parameters must be provided, and the new matrix will be
        initialized with the specified portion of the provided matrix.
        """
        self.data = None
        self.attr_names = []
        self.str_to_enum = []       # array of dictionaries
        self.enum_to_str = []       # array of dictionaries
        self.dataset_name = "Untitled"
        self.MISSING = float("infinity")

        if arff:
            self.load_arff(arff)
        elif matrix:
            if row_start is None:
                row_start = 0
            if col_start is None:
                col_start = 0
            if row_count is None:
                row_count = matrix.rows
            if col_count is None:
                col_count = matrix.cols
            self.init_from(matrix, row_start, col_start, row_count, col_count)
        else:
            pass
            # Confusion matrix needs to be initialized but have nothing in it
        
    def init_from(self, matrix, row_start, col_start, row_count, col_count):
        """Initialize the matrix with a portion of another matrix"""
        self.data = matrix.data[row_start:row_start+row_count, col_start:col_start+col_count]
        self.attr_names = matrix.attr_names[col_start:col_start+col_count]
        self.str_to_enum = matrix.str_to_enum[col_start:col_start+col_count]
        self.enum_to_str = matrix.enum_to_str[col_start:col_start+col_count]
        return self

    def add(self, matrix, row_start, col_start, col_count):
        """Appends a copy of the specified portion of a matrix to this matrix"""
        if self.cols != col_count:
            raise Exception("Incompatible number of columns")

        for col in range(self.cols):
            if matrix.value_count(col_start + col) != self.value_count(col):
                raise Exception("Incompatible relations")

        self.data = np.vstack((self.data, matrix.data[row_start:, col_start:col_start + col_count]))

    def set_size(self, rows, cols):
        """Resize this matrix (and set all attributes to be continuous)"""
        self.data = np.zeros((rows, cols))
        self.attr_names = [""] * cols
        self.str_to_enum = {}
        self.enum_to_str = {}

    def load_arff(self, filename):
        """Load matrix from an ARFF file"""
        self.data = None
        self.attr_names = []
        self.str_to_enum = []
        self.enum_to_str = []
        reading_data = False

        rows = []           # we read data into array of rows, then convert into array of columns

        f = open(filename)
        for line in f.readlines():
            line = line.rstrip()
            if len(line) > 0 and line[0] != '%':
                if not reading_data:
                    if line.lower().startswith("@relation"):
                        self.dataset_name = line[9:].strip()
                    elif line.lower().startswith("@attribute"):
                        attr_def = line[10:].strip()
                        if attr_def[0] == "'":
                            attr_def = attr_def[1:]
                            attr_name = attr_def[:attr_def.index("'")]
                            attr_def = attr_def[attr_def.index("'")+1:].strip()
                        else:
                            search = re.search(r'(\w*)\s*(.*)', attr_def)
                            attr_name = search.group(1)
                            attr_def = search.group(2)
                            # Remove white space from atribute values
                            attr_def = "".join(attr_def.split())

                        self.attr_names += [attr_name]

                        str_to_enum = {}
                        enum_to_str = {}
                        if not(attr_def.lower() == "real" or attr_def.lower() == "continuous" or attr_def.lower() == "integer"):
                            # attribute is discrete
                            assert attr_def[0] == '{' and attr_def[-1] == '}'
                            attr_def = attr_def[1:-1]
                            attr_vals = attr_def.split(",")
                            val_idx = 0
                            for val in attr_vals:
                                val = val.strip()
                                enum_to_str[val_idx] = val
                                str_to_enum[val] = val_idx
                                val_idx += 1

                        self.enum_to_str.append(enum_to_str)
                        self.str_to_enum.append(str_to_enum)

                    elif line.lower().startswith("@data"):
                        reading_data = True

                else:
                    # reading data
                    val_idx = 0
                    # print("{}".format(line))
                    vals = line.split(",")
                    row = np.zeros((len(vals)))
                    for val in vals:
                        val = val.strip()
                        if not val:
                            raise Exception("Missing data element in row with data '{}'".format(line))
                        else:
                            row[val_idx] = float(self.MISSING if val == "?" else self.str_to_enum[val_idx].get(val, val))

                        val_idx += 1

                    rows += [row]


        f.close()
        self.data = np.array(rows)

    @property
    def rows(self):
        """Get the number of rows in the matrix"""
        return self.data.shape[0]

    @property
    def cols(self):
        """Get the number of columns (or attributes) in the matrix"""
        return self.data.shape[1]

    def row(self, n):
        """Get the specified row"""
        return self.data[n]

    def col(self, n):
        """Get the specified column"""
        return self.data[:,n]

    def get(self, row, col):
        """
        Get the element at the specified row and column
        :rtype: float
        """
        return self.data[row, col]

    def set(self, row, col, val):
        """Set the value at the specified row and column"""
        self.data[row, col] = val

    def attr_name(self, col):
        """Get the name of the specified attribute"""
        return self.attr_names[col]

    def set_attr_name(self, col, name):
        """Set the name of the specified attribute"""
        self.attr_names[col] = name

    def attr_value(self, attr, val):
        """
        Get the name of the specified value (attr is a column index)
        :param attr: index of the column
        :param val: index of the value in the column attribute list
        :return:
        """
        return self.enum_to_str[attr][val]

    def value_count(self, col):
        """
        Get the number of values associated with the specified attribute (or columnn)
        0=continuous, 2=binary, 3=trinary, etc.
        """
        return len(self.enum_to_str[col]) if len(self.enum_to_str) > 0 else 0

    def shuffle(self, buddy=None):
        """Shuffle the row order. If a buddy Matrix is provided, it will be shuffled in the same order."""
        if not buddy:
          np.random.shuffle(self.data)
        else:
          if (self.data.shape[0] != buddy.data.shape[0]):
              raise Exception
          temp = np.hstack((self.data, buddy.data))
          np.random.shuffle(temp)
          self.data, buddy.data = temp[:,:self.cols], temp[:,self.cols:]

    def column_mean(self, col):
        """Get the mean of the specified column"""

        col_data = self.col(col)
        return np.mean(col_data[np.isfinite(col_data)])

    def column_min(self, col):
        """Get the min value in the specified column"""
        col_data = self.col(col)
        return np.min(col_data[np.isfinite(col_data)])

    def column_max(self, col):
        """Get the max value in the specified column"""
        col_data = self.col(col)
        return np.max(col_data[np.isfinite(col_data)])

    def most_common_value(self, col):
        """Get the most common value in the specified column"""
        col_data = self.col(col)
        (val, count) = stats.mode(col_data[np.isfinite(col_data)])
        return val[0]

    def normalize(self):
        """Normalize each column of continuous values"""
        for i in range(self.cols):
            if self.value_count(i) == 0:     # is continuous
                min_val = self.column_min(i)
                max_val = self.column_max(i)
                self.data[:,i] = (self.data[:,i] - min_val) / (max_val - min_val)

    def get_matrix_as_arff(self):
        """ Get representation of matrix as arff-style string
            Returns:
                string
        """
        out_string = ""
        out_string += "@RELATION {}".format(self.dataset_name)+ "\n"
        for i in range(len(self.attr_names)):
            out_string += "@ATTRIBUTE {}".format(self.attr_names[i])
            if self.value_count(i) == 0:
                out_string += (" CONTINUOUS")+ "\n"
            else:
                out_string += (" {{{}}}".format(", ".join(self.enum_to_str[i].values())))+ "\n"

        out_string += ("@DATA")+ "\n"
        for i in range(self.rows):
            r = self.row(i)

            values = []
            for j in range(len(r)):
                if self.value_count(j) == 0:
                    values.append(str(r[j]))
                else:
                    values.append(self.enum_to_str[j][r[j]])

            # values = list(map(lambda j: str(r[j]) if self.value_count(j) == 0 else self.enum_to_str[j][r[j]],
            #                   range(len(r))))
            out_string += ("{}".format(", ".join(values)))+ "\n"

        return out_string

    def __str__(self):
        return self.get_matrix_as_arff()

    def print(self):
        print(self)

    def nd_array(self, obj):
        """ Convert a matrix, list, or numpy array to numpy array
        Args:
            obj (array-like): An object to be converted
        Returns
            numpy array
        """

        if isinstance(obj, Matrix):
            return obj.data
        elif isinstance(obj, list):
            return np.ndarray(obj)
        elif isinstance(obj, np.ndarray):
            return obj
        else:
            raise Exception("Unrecognized data type")

    def append_columns(self, columns):
        """ Add columns to matrix. Number of rows must match existing Matrix.
        Args:
            columns (array-like): columns can be a Matrix, numpy array, or list
        """
        columns_to_add = self.nd_array(columns)
        if self.rows != columns_to_add.shape[0]:
            raise Exception("Incompatible number of rows: {}, {}".format(self.rows,columns_to_add.shape[0]))
        self.data = np.concatenate([self.data, columns_to_add], axis=1)
        return self

    def append_rows(self, rows):
        """ Add rows to matrix. Number of columns must match existing Matrix.
        Args:
            rows (array-like): rows can be a Matrix, numpy array, or list
        """
        rows_to_add = self.nd_array(rows)
        if self.cols != rows_to_add.shape[1]:
            raise Exception("Incompatible number of columns: {}, {}".format(self.cols, rows_to_add.shape[1]))
        self.data = np.concatenate([self.data, rows_to_add],axis=0)
        return self

