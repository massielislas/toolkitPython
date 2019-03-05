from __future__ import (absolute_import, division, print_function, unicode_literals)
from supervised_learner import SupervisedLearner
from arff import Arff
from decision_tree_node import TreeNode

import pandas as pd

import numpy as np
import math


class DecisionTreeLearner(SupervisedLearner):
    """
    For nominal labels, this model simply returns the majority class. For
    continuous labels, it returns the mean value.
    If the learning model you're using doesn't do as well as this one,
    it's time to find a new learning model.
    """

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
        This function should loop through the data and create/update a ML model until some stopping criteria is reached
        Args:
            features (Arff): 2D array of feature values (all instances)
            labels (Arff): 2D array of feature labels (all instances)
        """

        # print("FEATURES", features)
        # print("LABELS", labels)

        # pick the first one
        # all_decisions = [i for i in range(features.cols)]
        # parent_node = TreeNode()
        # parent_node.data_n = labels.rows
        # self.nodes += [parent_node]
        # output_classes_num = labels.value_count(0)
        # print("OUTPUT CLASSES", output_classes_num)


        all_decisions = [i for i in range(len(features[0]))]
        parent_node = TreeNode()
        parent_node.data_n = labels.instance_count
        output_classes_num = labels.unique_value_count(0)

        print("OUTPUT CLASSES", output_classes_num)
        class_counts = []

        last_decision_made = parent_node

        # # GET INFORMATION FOR (FIRST )NODE
        # for class_num in range(output_classes_num):
        #     class_counts += [labels.col(0).count(class_num)]
        #     class_count = labels.col(0).count(class_num)
        #     last_decision_made.information += ((-1) * class_count / last_decision_made.data_n) * math.log(class_count / last_decision_made.data_n, 2)
        #

        for class_num in range(output_classes_num):
            pre_split = labels.data[:, 0] == class_num
            outputs_of_class = labels.data[pre_split]
            class_count = len(outputs_of_class)
            last_decision_made.information += ((-1) * class_count / last_decision_made.data_n) * math.log(class_count / last_decision_made.data_n, 2)

        print('PARENT INFORMATION', last_decision_made.information)


        ###########################################
        self.average_label = []
        for i in range(labels.shape[1]): # for each label column
            if labels.is_nominal(i): # assumes 1D label
                self.average_label += [labels.most_common_value(0)]    # nominal
            else:
                self.average_label += [labels.column_mean(0)]  # continuous

                class_counts = []




    def predict_all(self, features):
        """ Make a prediction for each instance in dataset
        Args:
            features (2D array-like): Array of feature values
        Returns:
            array-like: 2D array of predictions (shape = instances, # of output classes)
        """
        pred = np.tile(self.average_label, features.shape[0]) # make a 1D vector of predictions, 1 for each instance
        return pred.reshape(-1,1) # reshape this so it is the correct shape = instances, # of output classes

    def unique(self, list_x):
        set_x = set(list_x)
        return list(set_x)

