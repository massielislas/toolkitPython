from __future__ import (absolute_import, division, print_function, unicode_literals)

from unittest import TestCase,TestLoader,TextTestRunner
from toolkit.baseline_learner import BaselineLearner
from toolkit.arff import Arff

class TestBaselineLearner(TestCase):

    data = Arff(arff="../test/cm1_req.arff")
    features = Arff(data, 0, 0, data.rows, data.cols - 1)
    labels = Arff(data, 0, data.cols - 1, data.rows, 1)
    learner = BaselineLearner()

    def test_train(self):
        self.learner.train(self.features, self.labels)
        self.assertAlmostEqual(self.learner.average_label[0], 0.0, places=4)

    def test_predict(self):
        self.learner.train(self.features, self.labels)
        label = self.learner.predict(self.features.row(0))

        self.assertListEqual(self.learner.average_label, label)

suite = TestLoader().loadTestsFromTestCase(TestBaselineLearner)
TextTestRunner(verbosity=2).run(suite)
