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

    def __init__(self, data=None, example_hyperparameter=None):
        pass


    def train(self, features, labels):
        self.labels = labels
        self.features = features


    def predict_all(self, features):
        """
        :type features: Arff
        """
        most_common_label = self.labels.most_common_value(0)

        predictions_list = []

        closes_indices = []

        # go through all instances there are to predict
        for examining_row_n, examining_row in enumerate(features.data):
            # print('EXAMINING ROW', examining_row)
            closest_labels = [most_common_label for i in range(self.k)]
            closest_distances = [sys.maxsize for i in range(self.k)]
            closes_indices = [-1 for i in range(self.k)]

            # go through all the features there are to compare against
            for i, row_to_compare in enumerate(self.features.data):
                if np.array_equal(examining_row, row_to_compare) is not True:
                    # print('ROW TO COMPARE', row_to_compare)
                    distance_between_instances = 0

                    # go through all attributes of an instance
                    for j in range(len(self.features.data[i])):
                        # print('j', j)
                        if features.is_nominal(j):
                            if examining_row[j] != row_to_compare[j]:
                                distance_between_instances += 1

                        else:
                            # print("ROW", i)
                            # print('COLUMN',j)
                            # print('VALUE 1', examining_row[j])
                            # print('VALUE 2', row_to_compare[j])
                            distance_difference = (examining_row[j] - row_to_compare[j]) ** 2
                            distance_between_instances += distance_difference

                    # if this is one of the "closest so far" instances
                    # print('DISTANCE', distance_between_instances ** .5)
                    distance_between_instances = distance_between_instances ** .5
                    #print('INITIAL', closest_distances)
                    if max(closest_distances) > distance_between_instances:
                        largest_distance = np.argmax(closest_distances)
                        closest_distances[largest_distance] = distance_between_instances

                        closest_labels[largest_distance] = self.labels.data[i][0]
                        closes_indices[largest_distance] = i
                        print('SUB!!!', i)
                    #print('EVALUATE', clo)
                # else:
                #     print('These two are the same!!!!', row_to_compare, examining_row)

            # votes dictionary
            # print('CLOSES LABELS', closest_labels)
            print('CLOSEST DISTANCES', closest_distances)
            print('CLOSEST LABELS', closest_labels)
            print('CLOSEST INDICES', closes_indices)
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
                    if votes[label] is None:
                        votes[label] = 0

                # print('VOTES DURING', votes)

                votes[label] += vote

            print('VOTES AFTER', votes)
            prediction = max(votes.items(), key=operator.itemgetter(1))[0]
            predictions_list += [prediction]

        print('PREDICTIONS LIST', predictions_list)



        # print(np.bincount(x).argmax())
        predictions = np.asarray(predictions_list, dtype=np.float64)
        predictions_return = predictions.reshape(-1, 1)
        return predictions_return


    def unique(self, list_x):
        if type(list_x) is list:
            set_x = set(list_x)
            return list(set_x)

        elif type(list_x) is np.ndarray:
            return np.unique(list_x)

        else:
            print('YOU NEED TO IMPLEMENT ANOTHER TYPE')

