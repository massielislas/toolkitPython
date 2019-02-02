from __future__ import (absolute_import, division, print_function, unicode_literals)

from unittest import TestCase,TestLoader,TextTestRunner
import numpy as np
import os
from toolkit import utils
from toolkit import baseline_learner, utils, manager, arff

class TestMatrix(TestCase):

    infinity = float("infinity")

    def setUp(self):
        import os
        #self.data = matrix.Matrix(arff="../test/cm1_req.arff")
        #self.features = matrix.Matrix(self.data, 0, 0, self.data.rows, self.data.cols - 1)
        self.my_manager = manager.MLSystemManager()
        self.arff_path = "../test/cm1_req.arff"

    def test_command_line_arguments(self):
        ## Create manager - from commandline argument
        args = r'-L baseline -A {} -E training'.format(self.arff_path)
        session = self.my_manager.create_session_from_argv(args)

    def test_function_arguments(self):
        my_learner = baseline_learner.BaselineLearner
        session = self.my_manager.create_new_session(arff_file=iris_data, learner=my_learner, eval_method="training",
                                                eval_parameter=None, print_confusion_matrix=False, normalize=False,
                                                random_seed=None)

        ## Create a Matrix object from arff
        iris = arff.Arff(arff=iris_data)

if __name__=="__main__":
    suite = TestLoader().loadTestsFromTestCase(TestMatrix)
    TextTestRunner(verbosity=2).run(suite)
