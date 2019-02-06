from __future__ import (absolute_import, division, print_function, unicode_literals)

from .supervised_learner import SupervisedLearner
from .baseline_learner import BaselineLearner
from .arff import Arff
import random
import argparse
import time
import sys
import textwrap
import inspect
import warnings
import numpy as np

""" IDEAS:

* Still support END-TO-END commandline stuff
* Also support: read in arff file, feed arff to learner, feed in learner to session
* ARFF class: keep track of feature types!!
* Can split ARFF files into features/labels; make sure this is done intelligently and appopriately

** Learner can be created at train time
** Learner can be created before, but then what does the session do?
** Session functions can be put into supervised learner

"""


class MLSystemManager:
    """ This class manages Toolkit sessions. Each session is initiated with a specific dataset (arff file), evaluation type (training, test, etc.), and learner.
    """

    def __init__(self):
        self.sessions = []
        self.doc_string = textwrap.dedent('''
                            ML toolkit manager
                            Usage: python toolkit.manager -L [learningAlgorithm] -A [ARFF_File] -E [EvaluationMethod] {[ExtraParamters]} [-N] [-R seed]

                            Possible evaluation methods are:
                            python toolkit.manager -L [learningAlgorithm] -A [ARFF_File] -E training
                            python toolkit.manager -L [learningAlgorithm] -A [ARFF_File] -E static [TestARFF_File]
                            python toolkit.manager -L [learningAlgorithm] -A [ARFF_File] -E random [PercentageForTraining]
                            python toolkit.manager -L [learningAlgorithm] -A [ARFF_File] -E cross [numOfFolds]''')

    def create_new_session(self, arff_file, learner, eval_method, eval_parameter=None, print_confusion_matrix=False, normalize=False, random_seed=None):
        """

        Args:
            arff_file (str): Path to arff file
            learner (learner): A "Learner" class object
            eval_method (str): String of evaluation method (training, static, random, )
            eval_parameter (str): Parameter for eval method
            print_confusion_matrix (bool): Whether to print confusion matrix for classification
            normalize (bool): whether to normalize data
            random_seed (int): Seed to initialize random number generator (e.g. to make results deterministic

        Returns:

        """
        new_session = ToolkitSession(arff_file=arff_file, learner=learner, eval_method=eval_method,
                                   eval_parameter=eval_parameter, print_confusion_matrix=print_confusion_matrix,
                                   normalize=normalize, random_seed=random_seed)
        self.sessions.append(new_session)
        return new_session

    def get_learner(self, model):
        """
        Get an instance of a learner for the given model name.

        To use toolkitPython as external package, you can extend this class (MLSystemManager)
        with your own custom class located outside of this package, and override this method
        to return your custom learners.

        :type model: str
        :rtype: SupervisedLearner
        """
        modelmap = {
            "baseline": BaselineLearner,
            # "perceptron": PerceptronLearner,
            # "mlp": MultilayerPerceptronLearner,
            # "decisiontree": DecisionTreeLearner,
            # "knn": InstanceBasedLearner
        }

        if model in modelmap:
            return modelmap[model]
        else:
            raise Exception("Unrecognized model: {}".format(model))

    def parser(self):
        parser = ToolkitArgParser(description="Usage: python toolkit.manager -L [learningAlgorithm]"+
                "-A [ARFF_File] -E [EvaluationMethod] {[ExtraParamters]} [-N] [-R seed]", doc_string=self.doc_string)

        parser.add_argument('-V', '--verbose', action='store_true', help='Print the confusion matrix and learner accuracy on individual class values')
        parser.add_argument('-N', '--normalize', action='store_true', help='Use normalized data')
        parser.add_argument('-R', '--seed', help="Random seed") # will give a string
        parser.add_argument('-L', required=True, choices=['baseline', 'perceptron', 'mlp', 'decisiontree', 'knn'], help='Learning Algorithm')
        parser.add_argument('-A', '--arff', metavar='filename', required=True, help='ARFF file')
        parser.add_argument('-E', metavar=('METHOD', 'args'), required=True, nargs='+', help="Evaluation method (training | static <test_ARFF_file> | random <%%_for_training> | cross <num_folds>)")
        return parser

    def create_session_from_argv(self, args=None):
        """ Parses command line arguments and creates the appropriate session
        """
        if len(sys.argv) > 1 and not args is None:
            #raise Exception("Cannot specify both argv and arguments in function call.")
            warnings.warn("Argv and function call arguments detected, defaulting to function call arguments.")            
        
        if not args is None:
            import shlex
            args = self.parser().parse_args(shlex.split(args))
        else:
            args = self.parser().parse_args()
            
        file_name = args.arff
        learner_name = args.L
        learner = self.get_learner(learner_name)
        eval_method = args.E[0]
        eval_parameter = args.E[1] if len(args.E) > 1 else None
        print_confusion_matrix = args.verbose
        normalize = args.normalize
        seed = args.seed  # Use a seed for deterministic results, if provided (makes debugging easier)

        new_session = self.create_new_session(arff_file=file_name, learner=learner, eval_method=eval_method,
                                   eval_parameter=eval_parameter, print_confusion_matrix=print_confusion_matrix,
                                   normalize=normalize, random_seed=seed)
        return new_session

class ToolkitSession:
    """ Toolkit session is given a learner with an associated arff file.
        Notes:
            * A learner class can be passed without instantiation. It will be created when the session is started.
                * Learner keyword arguments can be passed to the session
                * A learner class can also already be instantiated when passed
    """
    def __init__(self, arff_file, learner, eval_method=None, eval_parameter=None, print_confusion_matrix=False, normalize=False, random_seed=None, **kwargs):
        # parse the command-line arguments

        if random_seed:
            random.seed(random_seed)
            np.random.seed(random_seed)

        # update class variables
        if inspect.isclass(learner):
            # Instantiate learner if needed
            self.learner = learner(**kwargs)
            self.learner_name = learner.__name__
        else:
            self.learner = learner
            self.learner_name = type(learner).__name__


        self.print_confusion_matrix_flag = print_confusion_matrix
        self.eval_method = eval_method
        self.eval_parameter = eval_parameter
        self.normalize = normalize

        self.training_accuracy = []
        self.test_accuracy = []

        # load the ARFF file
        self.arff = Arff()
        self.arff.load_arff(arff_file)
        if self.normalize:
            print("Using normalized data")
            self.arff.normalize()

        # print some stats
        print("\nDataset name: {}\n"
              "Number of instances: {}\n"
              "Number of attributes: {}\n"
              "Learning algorithm: {}\n"
              "Evaluation method: {}\n".format(arff_file, self.arff.shape[0], self.arff.shape[1], self.learner_name, self.eval_method))

        if not eval_method is None:
            self.main()

    def main(self):
        if self.eval_method == "training":
            self.train(self.arff.get_features(), self.arff.get_labels(), self.print_confusion_matrix_flag )
        elif self.eval_method == "random":
            train_features, train_labels, test_features, test_labels = self.training_test_split(
                train_percent=self.eval_parameter)
            self.train(train_features, train_labels, self.print_confusion_matrix_flag)
            self.test(test_features, test_labels, self.print_confusion_matrix_flag)

        elif self.eval_method == "static":
            self.train(self.arff.get_features(), self.arff.get_labels(), self.print_confusion_matrix_flag)
            arff_file = self.eval_parameter
            test_data = Arff(arff_file)
            if self.normalize:
                test_data.normalize()
            self.test(features=test_data.get_features(), labels=test_data.get_labels(), print_confusion=self.print_confusion_matrix_flag)

        elif self.eval_method == "cross":
            self.cross_validate()

    def training_test_split(self, train_percent=.9):
        """ Arff object with labels included
        Args:
            train_percent:

        Returns:
            Tuple: train_features, train_labels, test_features, test_labels
        """
        self.arff.shuffle()

        print("Calculating accuracy on a random hold-out set...")
        train_percent = float(train_percent)
        if train_percent < 0 or train_percent > 1:
            raise Exception("Percentage for random evaluation must be between 0 and 1")
        print("Percentage used for training: {}".format(train_percent))
        print("Percentage used for testing: {}".format(1 - train_percent))

        train_size = int(train_percent * self.arff.shape[0])

        train_features = self.arff.get_features(slice(0, train_size))
        train_labels = self.arff.get_labels(slice(0, train_size))

        test_features = self.arff.get_features(slice(train_size, None))
        test_labels = self.arff.get_labels(slice(train_size, None))
        return train_features, train_labels, test_features, test_labels

    def train(self, features=None, labels=None, print_confusion=None):
        """By default, this trains on entire arff file. Features and labels options are given to e.g.
            train on only a part of the data
        Args:
            features (array-like):
            labels (array-like):
        Returns:

        """
        print("Calculating accuracy on training set...")

        if features is None:
            features = self.arff.get_features()
        if labels is None:
            labels = self.arff.get_labels()
        if print_confusion is None:
            print_confusion = self.print_confusion_matrix_flag

        confusion = None
        if print_confusion and labels.unique_value_count()>0:
            confusion = Arff(np.zeros([labels.unique_value_count(), labels.unique_value_count()]))

        start_time = time.time()
        self.learner.train(features, labels)
        elapsed_time = time.time() - start_time
        print("Time to train (in seconds): {}".format(elapsed_time))
        accuracy = self.learner.measure_accuracy(features, labels, confusion)
        self.training_accuracy.append(accuracy)
        print("Training set accuracy: " + str(accuracy))

        if print_confusion:
            self.print_confusion_matrix(confusion)

    def test(self, features, labels, print_confusion=None):
            """ This eval_method 1) creates a 'random' training/test split according to some user-specified percentage,
                            2) trains the data
                            3) reports training AND test accuracy
            """
            if print_confusion is None:
                print_confusion = self.print_confusion_matrix_flag

            confusion_matrix = Arff() if print_confusion else None
            test_accuracy = self.learner.measure_accuracy(features, labels, confusion_matrix)
            self.test_accuracy.append(test_accuracy)
            print("Test set accuracy: {}".format(test_accuracy))
            if print_confusion:
                self.print_confusion_matrix(confusion_matrix)

    def print_confusion_matrix(self, confusion_matrix):
            print("\nConfusion matrix: (Row=target value, Col=predicted value)")
            print(confusion_matrix.data)
            print("")

    def generate_fold(self, folds):
        for i in range(folds):
            start_test = int(i * self.arff.shape[0] / folds)
            end_test = int((i + 1) * self.arff.shape[0] / folds)

            train_features = self.arff.get_features()[np.r_[0:start_test, end_test:]]
            train_labels = self.arff.get_labels()[np.r_[0:start_test, end_test:]]

            test_features = self.arff.get_features()[start_test:end_test]
            test_labels = self.arff.get_labels()[start_test:end_test]
            yield train_features, train_labels, test_features, test_labels

    def cross_validate(self, reps, folds):
        print("Calculating accuracy using cross-validation...")

        folds = int(self.eval_parameter)
        if folds <= 0:
            raise Exception("Number of folds must be greater than 0")
        print("Number of folds: {}".format(folds))
        reps = 1
        sum_accuracy = 0.0
        elapsed_time = 0.0

        for rep_counter in range(reps):
            self.arff.shuffle()

            for fold_counter, train_features, train_labels, test_features, test_labels in enumerate(self.generate_fold(folds)):
                start_time = time.time()

                # Train model
                self.learner.train(train_features, train_labels)
                elapsed_time += time.time() - start_time
                training_accuracy = self.learner.measure_accuracy(train_features, train_labels)

                # Get test accuracy
                test_accuracy = self.learner.measure_accuracy(test_features, test_labels)
                sum_accuracy += test_accuracy
                print("Rep={}, Fold={}, Accuracy={}".format(rep_counter, fold_counter, test_accuracy))

                self.training_accuracy.append(training_accuracy)
                self.test_accuracy.append(test_accuracy)

                elapsed_time /= (reps * folds)
                print("Average time to train (in seconds): {}".format(elapsed_time))
                print("Mean accuracy={}".format(sum_accuracy / (reps * folds)))

            else:
                raise Exception("Unrecognized evaluation method '{}'".format(self.eval_method))

class ToolkitArgParser(argparse.ArgumentParser):
    def __init__(self, description=None, doc_string=None):
        super(ToolkitArgParser, self).__init__(description=description)
        self.doc_string = doc_string

    def error(self, message):
        sys.stderr.write('Error: %s\n' % message)
        print(self.doc_string)
        sys.exit()


if __name__ == '__main__':
    manager = MLSystemManager()
    session = manager.create_session_from_argv()
    session.main()

