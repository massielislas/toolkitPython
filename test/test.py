from __future__ import (absolute_import, division, print_function, unicode_literals)
from unittest import TestCase,TestLoader,TextTestRunner
import numpy as np
import os
from toolkit import baseline_learner, utils, manager, arff

class TestManager(TestCase):
    infinity = float("infinity")

    def setUp(self):

        self.arff_path = "../test/datasets/cm1_req.arff"

    def test_train(self):
        my_learner = baseline_learner.BaselineLearner
        session = manager.ToolkitSession(arff_file=self.arff_path, learner=my_learner, eval_method=None,
                                                eval_parameter=None, print_confusion_matrix=False, normalize=False,
                                                random_seed=None)

        session.train()
        self.assertAlmostEqual(session.training_accuracy[0],0.77528089)

    def test_train_test(self):
        my_learner = baseline_learner.BaselineLearner
        session = manager.ToolkitSession(arff_file=self.arff_path, learner=my_learner, eval_method=None,
                                                eval_parameter=None, print_confusion_matrix=True, normalize=False,
                                                random_seed=None)

        train_features, train_labels, test_features, test_labels = session.training_test_split(.7) # 70% training
        session.train(train_features, train_labels)
        session.test(test_features, test_labels)


    def test_train_test(self):
        my_learner = baseline_learner.BaselineLearner
        session = manager.ToolkitSession(arff_file=self.arff_path, learner=my_learner, eval_method="training",
                                                eval_parameter=None, print_confusion_matrix=False, normalize=False,
                                                random_seed=None)
        self.assertAlmostEqual(session.training_accuracy[0],0.77528089)

    def test_pass_learner_params(self):
        pass



if __name__=="__main__":
    print("here")
    suite = TestLoader().loadTestsFromTestCase(TestManager)

    TextTestRunner(verbosity=2).run(suite)
