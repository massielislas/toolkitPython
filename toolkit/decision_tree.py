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
    number_of_attribute_values = []

    prune = False
    validation = True

    validation_set = None
    validation_set_labels = None
    training_set = None
    training_set_labels = None
    test_set = None
    test_set_labels = None

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
        :type features: Arff
        """
        # print(features.data)

        # for i in range(len(features.data)):
        #     for j in range(len(features.data[0])):
        #         set_missing_to = features.unique_value_count(j)
        #         if features.is_missing(features.data[i][j]):
        #             features.data[i, j] = set_missing_to


        if self.validation == True:
            self.split_validation_and_training_and_test(labels, features)
        else:
            self.training_set = features
            self.training_set_labels = labels
        for j in range(len(features.data[0])):
            self.number_of_attribute_values += [features.unique_value_count(j)]

        columns_with_missing_values = []

        for i in range(len(features.data)):
            for j in range(len(features.data[0])):
                set_missing_to = features.unique_value_count(j)
                if features.is_missing(features.data[i][j]):
                    features.data[i, j] = set_missing_to
                    if j not in columns_with_missing_values:
                        columns_with_missing_values += [j]

        for column in columns_with_missing_values:
            self.number_of_attribute_values[column] += 1


        # print(self.number_of_attribute_values)


        # for i in range(5):
        #     print("")
        """
        This function should loop through the data and create/update a ML model until some stopping criteria is reached
        Args:
            features (Arff): 2D array of feature values (all instances)
            labels (Arff): 2D array of feature labels (all instances)
        """
        self.all_decisions = [i for i in range(len(self.training_set[0]))]
        self.root_node = TreeNode()
        self.root_node.set_features(self.training_set)
        self.root_node.labels = self.training_set_labels
        self.output_classes_num = labels.unique_value_count(0)

        self.compute_node_information(node=self.root_node)

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



        # print(test_predictions)

        for i in range(5):
            print()
        print('CALCULATING TEST ACCURACY...')
        test_predictions = self.predict_all(self.test_set)
        test_accuracy = self.calculate_accuracy(test_predictions, self.test_set_labels)
        print(test_accuracy)
        print('CALCULATING TRAINING ACCURACY')
        training_predictions = self.predict_all(self.training_set)
        training_accuracy = self.calculate_accuracy(training_predictions, self.training_set_labels)
        print(training_accuracy)

        if self.prune == True:
            self.prune_tree(self.root_node)

    def prune_tree(self, node, default_accuracy):
        node.pruned = True
        predictions = self.predict_pruned(self.validation_set)
        accuracy = self.calculate_accuracy(predictions, self.validation_set_labels)

        if accuracy + 1 < default_accuracy:
            node.pruned = False

        if len(node.children) == 0:
            return


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

        #print(labels.data[0][0])

    def compute_node_information(self, node):
        """
        :type node: TreeNode
        """
        for class_num in range(self.output_classes_num):
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

            attribute_values = [i for i in range(self.number_of_attribute_values[attribute])]
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
                return
            else:
                # print("SOMETHING WENT WRONG")
                node.set_classification_label(node.labels.most_common_value(0))

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


        # print('TYPE OF DATAAAAAAAAAAAAAAAA')
        # print(type(features.data[0][0]))

        for row_num, row in enumerate(data):
            # print('ROW', row)
            current_node = self.root_node
            while current_node.classification_label is None:
                any_child = current_node.children[0]
                data_point_attribute_value = row[any_child.feature_decided]
                # print()
                # print('')
                # print('NUMBER OF ATTRIBUTE VALUES', self.number_of_attribute_values)
                # print('FEATURE DECIDED', any_child.feature_decided)
                # print('FEATURE VALUE FOR DATA POINT', data_point_attribute_value)
                # print()

                moved_down_the_tree = False
                for child in current_node.children:
                    if data_point_attribute_value == child.feature_value_decision:
                        moved_down_the_tree = True
                        current_node = child
                        # current_node.to_string();
                        break

                if moved_down_the_tree == False:
                    most_data = -1
                    for child in current_node.children:
                        if child.features_n > most_data:
                            most_data = child.features_n
                            current_node = child



            # print('FINAL DECISION', current_node.classification_label)
            predictions_list += [current_node.classification_label]
            # print('BUILDING PREDICTIONS', predictions_list)
            # print()
            # print()

        predictions = np.asarray(predictions_list, dtype=np.float64)
        predictions_return = predictions.reshape(-1, 1)

        return predictions_return

    def predict_pruned(self, features):
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


        # print('TYPE OF DATAAAAAAAAAAAAAAAA')
        # print(type(features.data[0][0]))

        for row_num, row in enumerate(data):
            # print('ROW', row)
            current_node = self.root_node
            while current_node.classification_label is None or current_node.pruned == True:
                any_child = current_node.children[0]
                data_point_attribute_value = row[any_child.feature_decided]
                # print()
                # print('')
                # print('NUMBER OF ATTRIBUTE VALUES', self.number_of_attribute_values)
                # print('FEATURE DECIDED', any_child.feature_decided)
                # print('FEATURE VALUE FOR DATA POINT', data_point_attribute_value)
                # print()

                moved_down_the_tree = False
                for child in current_node.children:
                    if data_point_attribute_value == child.feature_value_decision:
                        moved_down_the_tree = True
                        current_node = child
                        # current_node.to_string();
                        break

                if moved_down_the_tree == False:
                    most_data = -1
                    for child in current_node.children:
                        if child.features_n > most_data:
                            most_data = child.features_n
                            current_node = child

            # print('FINAL DECISION', current_node.classification_label)
            if current_node.pruned == False:
                predictions_list += [current_node.classification_label]

            else:
                predictions_list += [current_node.parent_node.labels.most_common_value(0)]
            # print('BUILDING PREDICTIONS', predictions_list)
            # print()
            # print()


        #print(x)

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


    def split_validation_and_training_and_test(self, labels, features):
        """
        :type features: Arff
        :type features: Arff
        """
        labels.shuffle(features)

        # rows_to_keep = self.get_indices_by_boolean(pre_split)
        # columns_to_keep = slice(parent_node.features.data[0].size)
        # new_features = parent_node.features.create_subset_arff(rows_to_keep, columns_to_keep, 0)
        # attribute_value_node.set_features(new_features)
        # columns_to_keep = slice(parent_node.labels.data[0].size)
        # new_labels = parent_node.labels.create_subset_arff(rows_to_keep, columns_to_keep, 0)

        whole_set_size = len(features.data)
        test_size = int(whole_set_size // (100/20))

        training_and_validation_size = whole_set_size - test_size

        validation_size = int(training_and_validation_size // (100/10))

        training_size = training_and_validation_size - validation_size

        rows_for_test = [i for i in range(test_size)]
        rows_for_validation = [i + test_size for i in range(validation_size)]
        rows_for_training = [i + test_size + validation_size for i in range(training_size)]


        columns_for_labels = slice(labels.data[0].size)
        columns_for_features = slice(features.data[0].size)

        # CREATE TEST SET
        self.test_set = features.create_subset_arff(rows_for_test, columns_for_features, 0)
        self.test_set_labels = labels.create_subset_arff(rows_for_test, columns_for_labels, 0)

        self.validation_set = features.create_subset_arff(rows_for_validation, columns_for_features, 0)
        self.validation_set_labels = labels.create_subset_arff(rows_for_validation, columns_for_labels, 0)

        self.training_set = features.create_subset_arff(rows_for_training, columns_for_features, 0)
        self.training_set_labels = labels.create_subset_arff(rows_for_training, columns_for_labels, 0)


        # print('TEST SET')
        # print(self.test_set)
        # print('TEST SET LABELS')
        # print(self.test_set_labels)
        #
        # print('VALIDATION SET')
        # print(self.validation_set)
        # print('VALIDATION SET LABELS')
        # print(self.validation_set_labels)
        #
        # print('TRAINING SET')
        # print(self.training_set)
        # print('TRAINING SET LABELS')
        # print(self.training_set_labels)

        # print('test', rows_for_test)
        # print('validation', rows_for_validation)
        # print('training', rows_for_training)

        return None



