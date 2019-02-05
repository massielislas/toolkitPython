from __future__ import (absolute_import, division, print_function, unicode_literals)
from unittest import TestCase,TestLoader,TextTestRunner
import numpy as np
import os
from toolkit import baseline_learner, utils, manager, arff

class TestManager(TestCase):
    infinity = float("infinity")

    def setUp(self):

        self.arff_path = "../test/cm1_req.arff"

    def test_function_arguments(self):
        my_learner = baseline_learner.BaselineLearner
        session = manager.ToolkitSession(arff_file=self.arff_path, learner=my_learner, eval_method="training",
                                                eval_parameter=None, print_confusion_matrix=False, normalize=False,
                                                random_seed=None)
        ## Create a Matrix object from arff
        iris = arff.Arff(arff=self.arff_path)

if __name__=="__main__":
    print("here")
    suite = TestLoader().loadTestsFromTestCase(TestManager)

    TextTestRunner(verbosity=2).run(suite)
