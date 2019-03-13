from __future__ import (absolute_import, division, print_function, unicode_literals)
from supervised_learner import SupervisedLearner
from arff import Arff

import numpy as np
import operator


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

        predictions_list = []

        # go through all instances there are to predict
        for examining_row_n, examining_row in enumerate(features.data):
            closest_labels = [None for i in range(self.k)]
            closest_distances = [-1 for i in range(self.k)]

            # go through all the features there are to compare against
            for i, row_to_compare in enumerate(self.features.data):
                distance_between_instances = 0

                # go through all attributes of an instance
                for j in range(len(self.labels.data[i])):

                    if examining_row[j] == row_to_compare[i][j]:
                        distance_between_instances += 1

                # if this is one of the "closest so far" instances
                if min(closest_distances) > distance_between_instances:
                    largest_distance = np.argmax(closest_distances)
                    closest_distances[largest_distance] = distance_between_instances

                    closest_labels[largest_distance] = self.labels.data[i][0]

            # votes dictionary
            votes = dict()
            for label_num, label in enumerate(closest_labels):

                if self.distance_weighting is True:
                    vote = 1 / closest_distances[label_num]**2

                else:
                    vote = 1

                votes[label] += vote
            prediction = max(votes.items(), key=operator.itemgetter(1))[0]
            predictions_list += prediction

        print(predictions_list)



        # print(np.bincount(x).argmax())
        # predictions = np.asarray(predictions_list, dtype=np.float64)
        # predictions_return = predictions.reshape(-1, 1)
        # return predictions_return



        pred = np.tile(self.average_label, features.shape[0]) # make a 1D vector of predictions, 1 for each instance
        return pred.reshape(-1,1) # reshape this so it is the correct shape = instances, # of output classes

