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
        for i in range(5):
            print("")
        """
        This function should loop through the data and create/update a ML model until some stopping criteria is reached
        Args:
            features (Arff): 2D array of feature values (all instances)
            labels (Arff): 2D array of feature labels (all instances)
        """
        all_decisions = [i for i in range(len(features[0]))]
        root_node = TreeNode()
        root_node.data = features.data
        root_node.set_data(features.data)
        root_node.labels = labels.data
        output_classes_num = labels.unique_value_count(0)

        print("OUTPUT CLASSES", output_classes_num)

        last_decision_made = root_node

        for class_num in range(output_classes_num):
            pre_split = labels.data[:, 0] == class_num
            outputs_of_class = labels.data[pre_split]

            # print(pre_split)
            # print(outputs_of_class)

            class_count = len(outputs_of_class)
            last_decision_made.information += ((-1) * class_count / last_decision_made.data_n) * math.log(class_count / last_decision_made.data_n, 2)

        print('PARENT INFORMATION', last_decision_made.information)

        decisions_to_make = self.sub_lists(all_decisions, root_node.decisions_made)

        print(all_decisions)
        print(root_node.decisions_made)
        print("DECISIONS TO MAKE", decisions_to_make)

        possible_next_decisions = []


        for attribute in decisions_to_make:
            print("ATTRIBUTE", attribute)
            possible_next_decisions_nodes = []

            attribute_values = [i for i in range(features.unique_value_count(attribute))]
            attribute_info_gain = 0
            # print('ATTRIBUTE VALUES', attribute_values)

            for attribute_value in attribute_values:
                print('ATTRIBUTE VALUE', attribute_value)
                attribute_value_node = TreeNode()
                possible_next_decisions_nodes += [attribute_value_node]

                attribute_value_node.add_decision(attribute)
                # print('ATTRIBUTE VALUE NODE DECISIONS', attribute_value_node.decisions_made)
                attribute_value_node.feature_value_decision = attribute_value

                pre_split = last_decision_made.data[:, attribute] == attribute_value
                attribute_value_node.set_data(last_decision_made.data[pre_split])
                attribute_value_node.labels = last_decision_made.labels[pre_split]
                attribute_info_gain += 0

                for output_class in range(output_classes_num):
                    pre_split_labels = attribute_value_node.labels[:,0] == output_class
                    output_class_num= len(attribute_value_node.labels[pre_split_labels])
                    print('OUTPUT CLASS NUM', output_class_num)

                attribute_info_gain *= attribute_value_node.data_n / last_decision_made.data_n
        #     for node in possible_next_decisions_nodes:
        #         node.information = attribute_info_gain
        #
        #     possible_next_decisions += [possible_next_decisions_nodes]

            #     attribute_node.feature_value_decision = attribute_value
            #     attribute_value_count = attribute_column.count(attribute_value)
            #     attribute_info_gain += 0
            #     # add calculted info gain

            ## attribute_info_gain = attribute_count
            # attribute_info_gains += [attribute_info_gain]


        ###########################################
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

    def unique(self, list_x):
        set_x = set(list_x)
        return list(set_x)


    def sub_lists(self, li1, li2):
        return (list(set(li1) - set(li2)))

