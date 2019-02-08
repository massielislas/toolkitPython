from __future__ import (absolute_import, division, print_function, unicode_literals)
from .supervised_learner import SupervisedLearner
from .arff import Arff

import numpy as np


class BaselineLearner(SupervisedLearner):
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
        #self. weights = np.random.random((size))
        self.average_label = []
        self.data_shape = data.shape if not data is None else None
        self.example_hyperparameter = example_hyperparameter

    def train(self, features, labels):
        """
        This function should loop through the data and create/update a ML model until some stopping criteria is reached
        :type features: Arff
        :type labels: Arff
        """
        self.average_label = []
        for i,label in enumerate(labels.T):
            if labels.is_nominal(i): # assumes 1D label
                self.average_label += [labels.most_common_value(0)]    # nominal
            else:
                self.average_label += [labels.column_mean(0)]  # continuous

    def predict(self, features):
        """
        This function runs 1 instance through the model and returns the model's predicted label
        :type features: [float]
        """
        return self.average_label



