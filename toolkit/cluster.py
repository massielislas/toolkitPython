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
    clusters_to_make = 4
    clusters = []
    centroids = []
    clusters_sse = []
    total_sse = 0
    # average_distance_between_nodes = 0
    stop_clustering = False
    experiment = False
    all_distances = np.zeros(1)
    percentile = 0

    # complete_link = False

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
            # print("COLUMN", col_n)
            if features.unique_value_count(col_n) != 0:
                categorical_columns += [col_n]

        self.clusters = np.arange(len(features.data))

        self.clusters = [[i] for i in range(len(features.data))]

        # print("CLUSTERS ", self.clusters)
        """
        This function should loop through the data and create/update a ML model until some stopping criteria is reached
        Args:
            features (Arff): 2D array of feature values (all instances)
            labels (Arff): 2D array of feature labels (all instances)
        """

        self.features = features
        data_n = len(features.data)

        # np.zeros((len(vals)))

        # print("DATA_N ", data_n)

        # print(np.zeros(data_n))

        if self.HAC is True:

            # CALCULATING ORIGINAL DISTANCE MATRIX
            self.original_matrix = np.zeros((data_n, data_n))
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
                # print("DISTANCES", distances)

                # self.average_distance_between_nodes += np.sum(distances)
                self.all_distances = np.append(self.all_distances, distances, axis=0)

                distances[examining_row_n] = np.Infinity
                self.original_matrix[examining_row_n] = distances

            # DONE CALCULATING DISTANCE MATRIX

            # REMOVE ZEROES FROM DISTANCES
            non_zeroes = (self.all_distances != 0)
            self.all_distances = self.all_distances[non_zeroes]

            self.all_distances = np.sort(self.all_distances)
            print("FINAL DISTANCES SORTED", self.all_distances)
            percentile_index = len(self.all_distances) // (100/50)
            print("75TH PERCENTILE INDEX", percentile_index)

            self.percentile = self.all_distances[int(percentile_index)]

            print("75TH PERCENTILE", self.percentile)

            changing_matrix = cp.deepcopy(self.original_matrix)


            while len(changing_matrix) != self.clusters_to_make:

                lowest_coordinates = np.unravel_index(changing_matrix.argmin(), changing_matrix.shape)

                # print("LOWEST COORDINATES", lowest_coordinates)
                distance_found = changing_matrix[lowest_coordinates[0]][lowest_coordinates[1]]
                print("DISTANCE FOUND", distance_found)
                if distance_found >= self.percentile and self.experiment is True:
                    # self.stop_clustering = True
                    break

                # KEEPING TRACK OF CLUSTERS
                self.clusters[lowest_coordinates[0]] += self.clusters[lowest_coordinates[1]]
                del self.clusters[lowest_coordinates[1]]
                new_matrix = np.delete(changing_matrix, lowest_coordinates[1], axis=1)
                new_matrix = np.delete(new_matrix, lowest_coordinates[1], axis=0)

                for col_n in range(len(new_matrix[lowest_coordinates[0]])):

                    # print("COLUMN NUMBER", col_n)

                    if col_n == lowest_coordinates[0]:
                        list_for_replacement = [np.Infinity, np.Infinity]

                    elif col_n < lowest_coordinates[1]:
                        list_for_replacement = [changing_matrix[lowest_coordinates[0]][col_n], changing_matrix[lowest_coordinates[1]][col_n]]

                    elif col_n >= lowest_coordinates[1]:
                        list_for_replacement = [changing_matrix[lowest_coordinates[0]][col_n + 1], changing_matrix[lowest_coordinates[1]][col_n + 1]]

                    if self.single_link is True:
                        replace_with = min(list_for_replacement)

                    else:
                        replace_with = max(list_for_replacement)

                    new_matrix[lowest_coordinates[0]][col_n] = replace_with
                    new_matrix[col_n][lowest_coordinates[0]] = replace_with

                changing_matrix = cp.deepcopy(new_matrix)

                if self.stop_clustering is True:
                    break
                # print("REDUCED MATRIX")
                # print(changing_matrix)

            self.average_label = []
            for i in range(labels.shape[1]): # for each label column
                if labels.is_nominal(i): # assumes 1D label
                    self.average_label += [labels.most_common_value(0)]    # nominal
                else:
                    self.average_label += [labels.column_mean(0)]  # continuous


        print()
        print("CLUSTERS")
        print(self.clusters)
        print("\n\n")

        print("NUMBER OF CLUSTERS", len(self.clusters))

        self.calculate_errors()


    def predict_all(self, features):
        """ Make a prediction for each instance in dataset
        Args:
            features (2D array-like): Array of feature values
        Returns:
            array-like: 2D array of predictions (shape = instances, # of output classes)
        """
        pred = np.tile(self.average_label, features.shape[0]) # make a 1D vector of predictions, 1 for each instance
        return pred.reshape(-1,1) # reshape this so it is the correct shape = instances, # of output classes

    def calculate_errors(self):
        self.centroids = []
        self.clusters_sse = []
        self.total_sse = 0
        self.calculate_centroids()
        self.calculate_clusters_SSEs()
        self.calculate_clustering_SSE()


    def calculate_centroids(self):

        # print("CALCULATING CENTROIDS")

        # print("CLUSTERS: ", self.clusters)

        self.centroids = []
        # print("CENTROIDS BEFORE", self.centroids)

        # print("CENTROIDS")

        for cluster in self.clusters:
            centroid = np.zeros(len(self.features.data[0]))
            # print("CENTROIDS IN LOOP", self.centroids)
            # print()
            # print("NEW CLUSTER")
            for data_point_n in cluster:
                # print("CENTROID", centroid)
                # print("DATA POINT IN CLUSTER", self.features.data[data_point_n])
                to_add = cp.deepcopy(self.features.data[data_point_n])
                nan_places = np.isnan(to_add)
                to_add[nan_places] = 0
                # print(to_add)
                centroid = np.add(centroid, to_add)
            centroid = np.divide(centroid, len(cluster))
            self.centroids += [cp.deepcopy(centroid)]
            # print(centroid)

        # print("TEST", self.centroids[0][0])


    def calculate_clusters_SSEs(self):
        for cluster_n, cluster in enumerate(self.clusters):
            cluster_sse = 0
            for data_point in cluster:
                centroid = self.centroids[cluster_n]
                subtracted = np.subtract(centroid, self.features.data[data_point])

                squared = np.square(subtracted)
                # print('SQUARED', squared)
                # print("SQUARED B")
                # print(squared)
                nan_places = np.isnan(squared)
                squared[nan_places] = 1
                # print("SQUARED A")
                # print(squared)
                # print()
                # squared = np.square(subtracted)
                summed = np.sum(squared)
                # distance = np.sqrt(summed)
                # print("DISTANCE", distance)

                cluster_sse += summed

            # print("CLUSTER SSE", cluster_sse)
            self.clusters_sse += [cluster_sse]

        print("CLUSTERS SSE", self.clusters_sse)

                # pass

    def calculate_clustering_SSE(self):
        for sse in self.clusters_sse:
            self.total_sse += sse

        print("TOTAL SSE", self.total_sse)


# squared = np.square(subtracted)
#                 nan_places = np.isnan(squared)
#                 squared[nan_places] = 1
#
#                 summed = np.sum(squared, axis=1)
#
#                 distances = np.sqrt(summed)
