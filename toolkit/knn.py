from __future__ import (absolute_import, division, print_function, unicode_literals)
from supervised_learner import SupervisedLearner
from arff import Arff

import numpy as np
import operator
import copy as cp


class InstanceBasedLearner(SupervisedLearner):

    labels = None
    features = None
    k = 13
    distance_weighting = False
    continuous_output = False
    possible_missing_values = False
    experiment = True

    def __init__(self, data=None, example_hyperparameter=None):
        pass


    def train(self, features, labels):
        self.labels = labels
        self.features = features

        if self.experiment is True:
            self.reduction(features, labels)


        if labels.unique_value_count(0) == 0:
            self.continuous_output = True
            # print('SETTING CONTINUOUS OUTPUT TO TRUE')

        print('!!! K !!!', self.k)

    def reduction(self, features, labels):
        """
        :type features: Arff
        :type labels: Arff
        """

        # old_features = cp.deepcopy(self.features)
        # old_labels = cp.deepcopy(self.labels)
        old_accuracy = -1
        new_accuracy = 100
        counter = 0

        predictions = self.predict_all(features)
        baseline_accuracy = self.calculate_accuracy(predictions, labels)
        print('BASELINE ACCURACY', baseline_accuracy)

        while new_accuracy > baseline_accuracy - 1:
            print('REPLACE', counter)

            self.features.shuffle(self.labels)
            features.shuffle(labels)

            # old_accuracy = new_accuracy
            old_features = cp.deepcopy(self.features)
            old_labels = cp.deepcopy(self.labels)

            old_features.shuffle(old_labels)

            rows_to_keep = [i for i in range(len(old_features.data) - 1)]
            # print('ROWS TO KEEP', rows_to_keep)
            columns_to_keep = slice(old_features.data[0].size)
            new_features = old_features.create_subset_arff(rows_to_keep, columns_to_keep, 0)
            columns_to_keep_labels = slice(old_labels.data[0].size)
            new_labels = old_labels.create_subset_arff(rows_to_keep, columns_to_keep_labels, 0)

            self.labels = cp.deepcopy(new_labels)
            self.features = cp.deepcopy(new_features)

            predictions = self.predict_all(features)
            new_accuracy = self.calculate_accuracy(predictions, labels)

            print('ACCURACY', new_accuracy)
            counter += 1

        self.features = cp.deepcopy(old_features)
        self.labels = cp.deepcopy(old_labels)
        print('DONE REPLACING')

        # predictions = self.predict_all(features)
        # new_accuracy = self.calculate_accuracy(predictions, labels)


    def calculate_accuracy(self, predictions, labels):
        # print('PREDICTIONS')
        # print(predictions)
        # print('LABELS')
        # print(labels.data)
        correct_predictions = 0
        for i,label in enumerate(labels.data):
            # print('PREDICTIONS AND LABEL', predictions[i], label)
            if predictions[i] == label:
                correct_predictions += 1

        # print('LENGTH', len(predictions))
        # print('CORRECT PREDICTIONS')
        # print(correct_predictions)
        return correct_predictions / len(predictions)


    def predict_all(self, features):
        """
        :type features: Arff
        """

        if self.possible_missing_values is True:
            for row_num, row in enumerate(features.data):
                for col_num, col in enumerate(row):
                    if features.is_missing(col):
                        print('MISSING')
                        unique_value_count = features.unique_value_count(col_num)

                        if unique_value_count == 0:
                            features.data[row_num, col_num] = np.mean(features, axis=0)[col_num]

                        else:
                            features.data[row_num, col_num] = unique_value_count


        predictions_list = []

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
            nan_places = np.isnan(squared)
            squared[nan_places] = 1

            # print('SQUARED REPLACED', squared)

            summed = np.sum(squared, axis=1)
            # print('SUMMED', summed)

            closest_distances = np.sqrt(summed)
            # print('DISTANCES', closest_distances)

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
                        if closest_distances[label_num]**2 != 0:
                            vote = 1 / closest_distances[label_num]**2
                        else:
                            vote = 1

                    else:
                        vote = 1

                    votes[label] += vote

                prediction = max(votes.items(), key=operator.itemgetter(1))[0]
                predictions_list += [prediction]

            elif self.continuous_output is True:
                # print('CALCULATING FOR CONTINUOUS OUTPUT')
                regression_value = 0
                distance_weights = 0
                for label_num, label in enumerate(closest_labels):
                    value_to_add = label
                    if self.distance_weighting is True:
                        distance_weight = closest_distances[label_num]**2
                        if distance_weight != 0:
                            value_to_add = value_to_add / distance_weight
                            distance_weights += 1 /distance_weight

                    regression_value += value_to_add

                if self.distance_weighting is True:
                    regression_value /= distance_weights
                else:
                    regression_value /= len(closest_labels)

                predictions_list += [regression_value]

            # print('VOTES AFTER', votes)


        # print('PREDICTIONS LIST', predictions_list)



        # print(np.bincount(x).argmax())
        # print(predictions_list)
        predictions = np.asarray(predictions_list, dtype=np.float64)
        predictions_return = predictions.reshape(-1, 1)
        # print('TRANSFORMED', predictions)
        # print(self.labels.data)
        print('DATA LENGTH', len(self.labels.data))
        print('LIST LENGTH', len(predictions_list))
        return predictions_return


    def unique(self, list_x):
        if type(list_x) is list:
            set_x = set(list_x)
            return list(set_x)

        elif isinstance(list_x, np.ndarray):
            return np.unique(list_x)

        else:
            print('YOU NEED TO IMPLEMENT ANOTHER TYPE')

    def normalize(self, arff):
        """Normalize each column of continuous values"""
        for i in range(arff.shape[1]):
            if arff.unique_value_count(i) == 0:  # is continuous
                min_val = arff.column_min(i)
                max_val = arff.column_max(i)
                arff.data[:, i] = (arff.data[:, i] - min_val) / (max_val - min_val)

