from __future__ import (absolute_import, division, print_function, unicode_literals)
from .arff import Arff
import math
import numpy as np
import warnings
# this is an abstract class


class SupervisedLearner:

    def train(self, features, labels):
        """
        Before you call this method, you need to divide your data
        into a feature matrix and a label matrix.
        :type features: Arff
        :type labels: Arff
        """
        raise NotImplementedError()

    def predict(self, features):
        """
        A feature vector goes in. A label vector comes out. (Some supervised
        learning algorithms only support one-dimensional label vectors. Some
        support multi-dimensional label vectors.)
        :type features: [float]
        :type labels: [float]
        """
        raise NotImplementedError

    def measure_accuracy(self, features, labels, confusion=None, eval_method=None):
        """
        The model must be trained before you call this method. If the label is nominal,
        it returns the predictive accuracy. If the label is continuous, it returns
        the root mean squared error (RMSE). If confusion is non-NULL, and the
        output label is nominal, then confusion will hold stats for a confusion matrix.
        :type features: array-like
        :type labels: array-like
        :type confusion: Arff
        :rtype float
        """

        # If no eval_method is passed
        if eval_method==None:
            if isinstance(labels, Arff):
                if labels.is_nominal():
                    eval_method = "accuracy"
                else:
                    eval_method = "sse"
            elif isinstance(labels, np.ndarray):
                warnings.warn("Numpy array passed with no evaluation method, measuring accuracy")
                labels=Arff(labels)

        if features.shape[0] != labels.shape[0]:
            raise Exception("Expected the features and labels to have the same number of rows")
        if labels.shape[1] != 1:
            raise Exception("Sorry, this method currently only supports one-dimensional labels")
        if features.shape[0] == 0:
            raise Exception("Expected at least one row")

        if eval_method == "sse":
            # label is continuous
            pred = [0.0]
            sse = 0.0
            for i in range(features.shape[0]):
                feat = features[i]
                targ = labels[i]
                
                if len(pred > 0):
                    del pred[:]
                pred = self.predict(feat)
                delta = targ - pred
                sse += delta**2
            return math.sqrt(sse / features.shape[0])

        elif eval_method == "accuracy":

            # label is nominal, so measure predictive accuracy
            if confusion: # this assumes arff-class labels
                confusion.set_size(labels.unique_value_count(), labels.unique_value_count())
                confusion.attr_names = [labels.attr_value(0, i) for [i] in labels]

            correct_count = 0
            prediction = []
            for i in range(features.shape[0]):
                feat = features[i]
                targ = int(labels[i,0]) ## THIS ASSUME 1-D OUTPUTS

                if len(prediction) > 0:
                    del prediction[:]
                if targ >= labels.unique_value_count():
                    raise Exception("The label is out of range")

                # Assume predictions are integers 0-# of classes
                pred = np.asarray(self.predict(feat)).astype(int)[0] ## ASSUME 1-D prediction

                if confusion: # only working with one output?
                    confusion.data[targ,pred] += 1
                #print(pred,targ)
                if (pred == targ).all():
                    correct_count += 1

            return correct_count / features.shape[0]