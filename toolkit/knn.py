from __future__ import (absolute_import, division, print_function, unicode_literals)
from supervised_learner import SupervisedLearner
from arff import Arff

import numpy as np
import operator
import sys


class InstanceBasedLearner(SupervisedLearner):

    labels = None
    features = None
    k = 3
    distance_weighting = False
    continuous_output = False

    def __init__(self, data=None, example_hyperparameter=None):
        pass


    def train(self, features, labels):
        self.labels = labels
        self.features = features

        if labels.unique_value_count(0) == 0:
            self.continuous_output = True


    def predict_all(self, features):
        """
        :type features: Arff
        """
        most_common_label = self.labels.most_common_value(0)

        predictions_list = []

        closes_indices = []

        # print('BEFORE', self.features.data)

        # go through all instances there are to predict
        for examining_row_n, examining_row in enumerate(features.data):
            # print('EXAMINING ROW', examining_row)

            # print('BEFORE', self.features.data)
            # print("ROW", examining_row)
            subtracted = np.subtract(self.features.data, examining_row)
            # print('SUBTRACTED', subtracted)

            squared = np.square(subtracted)
            # print('SQUARED', squared)

            summed = np.sum(squared, axis=1)
            # print('SUMMED', summed)

            closest_distances = np.sqrt(summed)
            # print('DISTANCES', distances)

            sorted_smallest = np.argpartition(closest_distances, self.k+1)
            # print('SORTED SMALLEST', sorted_smallest)
            closest_labels = []

            for i in range(self.k + 1):
                closest_labels += [self.labels[sorted_smallest[i]][0]]

            # print('CLOSEST LABELS', closest_labels)

            if self.continuous_output is False:
                votes = dict()
                # print('VOTES BEFORE', votes)

                unique_labels = self.unique(closest_labels)

                for unique_label in unique_labels:
                    votes[unique_label] = 0

                for label_num, label in enumerate(closest_labels):
                    if self.distance_weighting is True:
                        vote = 1 / closest_distances[label_num]**2

                    else:
                        vote = 1

                    votes[label] += vote

            # print('VOTES AFTER', votes)
            prediction = max(votes.items(), key=operator.itemgetter(1))[0]
            predictions_list += [prediction]

        # print('PREDICTIONS LIST', predictions_list)



        # print(np.bincount(x).argmax())
        predictions = np.asarray(predictions_list, dtype=np.float64)
        predictions_return = predictions.reshape(-1, 1)
        return predictions_return


    def unique(self, list_x):
        if type(list_x) is list:
            set_x = set(list_x)
            return list(set_x)

        elif isinstance(list_x, np.ndarray):
            return np.unique(list_x)

        else:
            print('YOU NEED TO IMPLEMENT ANOTHER TYPE')

