from __future__ import (absolute_import, division, print_function, unicode_literals)
from supervised_learner import SupervisedLearner
from arff import Arff

import numpy as np
import copy as cp


class ClusterBasedLearner(SupervisedLearner):
    """
    For nominal labels, this model simply returns the majority class. For
    continuous labels, it returns the mean value.
    If the learning model you're using doesn't do as well as this one,
    it's time to find a new learning model.
    """

    original_matrix = None
    features = None
    HAC = True
    single_link = True
    clusters_to_make = 1

    def __init__(self, data=None, example_hyperparameter=None):
        """ Example learner initialization. Any additional variables passed to the Session will be passed on to the learner,
            e.g. learning rate, etc.

            Learners can initialize weights here
        Args:
            data:
            hyperparameters:
        """

        ## Example initializations - leave in for testing
        self.example_hyperparameter = example_hyperparameter
        self.data_shape = data.shape if not data is None else None
        # self.average_label = []
        # self. weights = np.random.random(self.data_shape)

    def train(self, features, labels):
        """
        :type features: Arff
        :type labels: Arff
        """
        categorical_columns = []

        for col_n in range(len(features.data[0])):
            print("COLUMN", col_n)
            if features.unique_value_count(col_n) != 0:
                categorical_columns += [col_n]
        """
        This function should loop through the data and create/update a ML model until some stopping criteria is reached
        Args:
            features (Arff): 2D array of feature values (all instances)
            labels (Arff): 2D array of feature labels (all instances)
        """

        self.features = features
        data_n = len(features.data)

        # np.zeros((len(vals)))

        print("DATA_N ", data_n)

        # print(np.zeros(data_n))

        if self.HAC is True:

            self.original_matrix = np.zeros((data_n, data_n))

            print(self.original_matrix)

            print("DATA", features.data)

            # calculating original matrix with distances between points
            for examining_row_n, examining_row in enumerate(features.data):
                subtracted = np.subtract(self.features.data, examining_row)

                for col in categorical_columns:
                    # >>> test[:,0]
                    # array([1, 3, 5])
                    np.place(subtracted[:, col], subtracted[:, col] != 0, [1])

                squared = np.square(subtracted)
                nan_places = np.isnan(squared)
                squared[nan_places] = 1

                summed = np.sum(squared, axis=1)

                distances = np.sqrt(summed)

                distances[examining_row_n] = np.Infinity

                print("DISTANCES")
                print(distances)
                print()
                self.original_matrix[examining_row_n] = distances


            print("CALCULATED MATRIX")
            print(self.original_matrix)


            changing_matrix = cp.deepcopy(self.original_matrix)

            while len(changing_matrix) != self.clusters_to_make:

                lowest_coordinates = np.unravel_index(changing_matrix.argmin(), changing_matrix.shape)

                print(lowest_coordinates)

                print("LOWEST COORDINATES")
                print(lowest_coordinates[0])
                print(lowest_coordinates[1])

                new_matrix = np.delete(changing_matrix, lowest_coordinates[1], axis=1)
                new_matrix = np.delete(new_matrix, lowest_coordinates[1], axis=0)

                print("CHANGING MATRIX")
                print(changing_matrix)

                print("NEW MATRIX")
                print(new_matrix)
                for col_n in range(len(new_matrix[lowest_coordinates[0]])):

                    # print("COLUMN NUMBER", col_n)

                    if col_n == lowest_coordinates[0]:
                        list_for_minimum = [np.Infinity, np.Infinity]
                        # print("DIAGONAL")

                    elif col_n < lowest_coordinates[1]:
                        list_for_minimum = [changing_matrix[lowest_coordinates[0]][col_n], changing_matrix[lowest_coordinates[1]][col_n]]
                        # print("LIST FOR MINIMUM", list_for_minimum)

                    elif col_n >= lowest_coordinates[1]:
                        list_for_minimum = [changing_matrix[lowest_coordinates[0]][col_n + 1], changing_matrix[lowest_coordinates[1]][col_n + 1]]
                        # print("COLUMN SWITCH")
                        # print("LIST FOR MINIMUM", list_for_minimum)

                    minimum = min(list_for_minimum)
                    new_matrix[lowest_coordinates[0]][col_n] = minimum
                    new_matrix[col_n][lowest_coordinates[0]] = minimum
                    # print("MINIMUM", new_matrix[lowest_coordinates[0]][col_n])

                changing_matrix = cp.deepcopy(new_matrix)
                print("REDUCED MATRIX")
                print(changing_matrix)


            self.average_label = []
            for i in range(labels.shape[1]): # for each label column
                if labels.is_nominal(i): # assumes 1D label
                    self.average_label += [labels.most_common_value(0)]    # nominal
                else:
                    self.average_label += [labels.column_mean(0)]  # continuous

    def predict_all(self, features):
        """ Make a prediction for each instance in dataset
        Args:
            features (2D array-like): Array of feature values
        Returns:
            array-like: 2D array of predictions (shape = instances, # of output classes)
        """
        pred = np.tile(self.average_label, features.shape[0]) # make a 1D vector of predictions, 1 for each instance
        return pred.reshape(-1,1) # reshape this so it is the correct shape = instances, # of output classes

