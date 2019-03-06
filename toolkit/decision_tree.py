from __future__ import (absolute_import, division, print_function, unicode_literals)
from supervised_learner import SupervisedLearner
from arff import Arff
from decision_tree_node import TreeNode


import numpy as np
import math
import copy as cp


class DecisionTreeLearner(SupervisedLearner):
    """
    For nominal labels, this model simply returns the majority class. For
    continuous labels, it returns the mean value.
    If the learning model you're using doesn't do as well as this one,
    it's time to find a new learning model.
    """

    all_decisions = []
    output_classes_num = 0
    root_node = None

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
        self.all_decisions = [i for i in range(len(features[0]))]
        self.root_node = TreeNode()
        self.root_node.set_features(features)
        self.root_node.labels = labels
        self.output_classes_num = labels.unique_value_count(0)

        print("OUTPUT CLASSES", self.output_classes_num)

        self.compute_node_information(node=self.root_node)
        print('PARENT INFORMATION', self.root_node.information)

        self.compute_children_of_node(self.root_node)

        for child in self.root_node.children:
            self.build_tree_recursive(child)

        print()
        # self.visualize_tree(self.root_node)
        self.average_label = []
        for i in range(labels.shape[1]): # for each label column
            if labels.is_nominal(i): # assumes 1D label
                self.average_label += [labels.most_common_value(0)]    # nominal
            else:
                self.average_label += [labels.column_mean(0)]  # continuous


    def compute_node_information(self, node):
        """
        :type node: TreeNode
        """
        for class_num in range(self.output_classes_num):
            # print("HEY LABELS", node.labels)
            pre_split = node.labels.data[:, 0] == class_num

            class_count = len(node.labels.data[pre_split])
            class_per_attribute = class_count / node.features_n
            # print("class per attribute, in calculating node info", class_per_attribute)
            if class_per_attribute != 0:
                node.information += ((-1) * class_per_attribute) * math.log(class_per_attribute, 2)
        # print('NODE INFORMATION', node.information)


    def compute_children_of_node(self, parent_node):
        """
        :type parent_node: TreeNode
        """
        decisions_to_make = self.sub_lists(self.all_decisions, parent_node.decisions_made)

        # print("DECISIONS MADE", parent_node.decisions_made)
        # print("LAST DECISION", parent_node.feature_value_decision)
        # print("DECISIONS TO MAKE", decisions_to_make)

        possible_next_decisions = []
        information_gains = []

        for attribute in decisions_to_make:
            # print("ATTRIBUTE", attribute)
            possible_next_decisions_nodes = []

            attribute_values = [i for i in range(parent_node.features.unique_value_count(attribute))]
            attribute_info_loss = 0

            for attribute_value in attribute_values:
                attribute_value_info_loss = 0
                # print('ATTRIBUTE VALUE', attribute_value)
                attribute_value_node = TreeNode()
                attribute_value_node.parent_node = parent_node

                attribute_value_node.add_decision(attribute)
                attribute_value_node.feature_value_decision = attribute_value
                # print('ATTRIBUTE VALUE NODE DECISIONS MADE', attribute_value_node.decisions_made)

                pre_split = parent_node.features.data[:, attribute] == attribute_value
                rows_to_keep = self.get_indices_by_boolean(pre_split)
                columns_to_keep = slice(parent_node.features.data[0].size)
                new_features = parent_node.features.create_subset_arff(rows_to_keep, columns_to_keep, 0)
                attribute_value_node.set_features(new_features)
                columns_to_keep = slice(parent_node.labels.data[0].size)
                new_labels = parent_node.labels.create_subset_arff(rows_to_keep, columns_to_keep, 0)
                attribute_value_node.labels = new_labels
                attribute_value_info_loss += 0

                if attribute_value_node.features_n > 0:
                    possible_next_decisions_nodes += [attribute_value_node]
                    for output_class in range(self.output_classes_num):
                        pre_split_labels = attribute_value_node.labels[:, 0] == output_class
                        class_count = len(attribute_value_node.labels[pre_split_labels])
                        # print('OUTPUT CLASS NUM', class_count)
                        # print('TOTAL IN ATTRIBUTE', attribute_value_node.features_n)
                        class_per_attribute = class_count / attribute_value_node.features_n
                        # print(class_per_attribute)
                        # print("CLASS COUNT", class_count)
                        if class_per_attribute != 0:
                            attribute_value_info_loss += (-1) * (class_count / attribute_value_node.features_n) * math.log(class_count / attribute_value_node.features_n, 2)

                attribute_value_info_loss *= attribute_value_node.features_n / parent_node.features_n
                attribute_info_loss += attribute_value_info_loss

            possible_next_decisions += [possible_next_decisions_nodes]
            # print('attribute info loss', attribute_info_loss)
            # print('parent information', parent_node.information)
            attribute_information_gain = parent_node.information - attribute_info_loss

            for child_node_created in possible_next_decisions_nodes:
                child_node_created.information_gain = attribute_information_gain

            information_gains += [attribute_information_gain]
            # print('ATTRIBUTE INFO LOSS', attribute_info_loss)
            # print('INFO GAIN FOR ATTRIBUTE', attribute_value_node.information_gain)

        # print("INFORMATION GAINS", information_gains)
        best_attribute = np.argmax(information_gains)
        # print("BEST ATTRIBUTE POSITION", best_attribute)
        parent_node.children = possible_next_decisions[best_attribute]
        # print('BEST ATTRIBUTE', possible_next_decisions[best_attribute][0].feature_decided)


    def build_tree_recursive(self, node):
        """
        :type node: TreeNode
        """

        decisions_to_make = self.sub_lists(self.all_decisions, node.decisions_made)
        unique_values = self.unique(node.labels.data)


        if len(decisions_to_make) == 0:
            # unique values will be a list
            unique_values = self.unique(node.labels.data)
            if len(unique_values) == 1:
                node.set_classification_label(unique_values[0])
                print('CLASSIFICATION LABEL', node.classification_label)
                return
            else:
                print("SOMETHING WENT WRONG")

        elif len(unique_values) == 1:
            node.set_classification_label(unique_values[0])
            # print('CLASSIFICATION LABEL', node.classification_label)
            # print('EVERRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR')
            return


        else:
            self.compute_node_information(node)
            self.compute_children_of_node(node)
            for child in node.children:
                self.build_tree_recursive(child)


    def predict_all(self, features):
        """ Make a prediction for each instance in dataset
        Args:
            features (2D array-like): Array of feature values
        Returns:
            array-like: 2D array of predictions (shape = instances, # of output classes)
        """
        data = cp.deepcopy(features.data)
        current_node = self.root_node
        classified = False
        predictions_list = []


        print('TYPE OF DATAAAAAAAAAAAAAAAA')
        print(type(features.data[0][0]))

        for row_num, row in enumerate(data):
            print('ROW', row)
            current_node = self.root_node
            while current_node.classification_label is None:
                any_child = current_node.children[0]
                data_point_attribute_value = row[any_child.feature_decided]
                print()
                print('FEATURE DECIDED', any_child.feature_decided)
                print('FEATURE VALUE FOR DATA POINT', data_point_attribute_value)
                print()

                for child in current_node.children:
                    if data_point_attribute_value == child.feature_value_decision:
                        current_node = child
                        current_node.to_string();
                        break

            print('FINAL DECISION', current_node.classification_label)
            predictions_list += [current_node.classification_label]
            print('BUILDING PREDICTIONS', predictions_list)
            print()
            print()



        pred = np.tile(self.average_label, features.shape[0]) # make a 1D vector of predictions, 1 for each instance
        x = pred.reshape(-1,1) # reshape this so it is the correct shape = instances, # of output classes

        print("X")
        print(x)
        print('FINAL PREDICTIONS')
        print(predictions_list)

        predictions = np.asarray(predictions_list, dtype=np.float64)

        return predictions.reshape(-1, 1)

    def unique(self, list_x):
        if type(list_x) is list:
            set_x = set(list_x)
            return list(set_x)

        elif type(list_x) is np.ndarray:
            return np.unique(list_x)

        else:
            print('YOU NEED TO IMPLEMENT ANOTHER TYPE')


    def sub_lists(self, li1, li2):
        return (list(set(li1) - set(li2)))

    def get_indices_by_boolean(self, list_of_booleans):
        indices_list = []
        for i in range(len(list_of_booleans)):
            if list_of_booleans[i] == True:
                indices_list += [i]

        return indices_list


    def visualize_tree(self, node):
        """
        :type node: TreeNode
        """
        if len(node.children) == 0:
            node.to_string()
            print()
            return

        else:
            node.to_string()
            print()
            for child in node.children:
                self.visualize_tree(child)


