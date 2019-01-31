from __future__ import (absolute_import, division, print_function, unicode_literals)

from .supervised_learner import SupervisedLearner
from .baseline_learner import BaselineLearner
from .matrix import Matrix
import random
import argparse
import time
import sys
import textwrap
import inspect
import warnings

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
        :param arff_file: (str) Path to arff file
        :param learner: (learner) A "Learner" class object
        :param eval_method: (str) String of evaluation method (training, static, random, )
        :param eval_parameter: (str) Parameter for eval method
        :param print_confusion_matrix: (bool) Whether to print confusion matrix for classification
        :param normalize: (bool) whether to normalize data
        :param random_seed: (int) Seed to initialize random number generator (e.g. to make results deterministic)
        :return:
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
            # "neuralnet": NeuralNetLearner,
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
        parser.add_argument('-L', required=True, choices=['baseline', 'perceptron', 'neuralnet', 'decisiontree', 'knn'], help='Learning Algorithm')
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
    def __init__(self, arff_file, learner, eval_method, eval_parameter=None, print_confusion_matrix=False, normalize=False, random_seed=None):
        # parse the command-line arguments


        if random_seed:
            random.seed(random_seed)

        # update class variables
        if inspect.isclass(learner):
            self.learner = learner()
            self.learner_name = learner.__name__
        else:
            self.learner = learner
            self.learner_name = type(learner).__name__
            
        self.print_confusion_matrix = print_confusion_matrix
        self.eval_method = eval_method
        self.eval_parameter = eval_parameter
        self.normalize = normalize

        # load the ARFF file
        self.data = Matrix()
        self.data.load_arff(arff_file)
        if self.normalize:
            print("Using normalized data")
            self.data.normalize()

        # print some stats
        print("\nDataset name: {}\n"
              "Number of instances: {}\n"
              "Number of attributes: {}\n"
              "Learning algorithm: {}\n"
              "Evaluation method: {}\n".format(arff_file, self.data.rows, self.data.cols, self.learner_name, self.eval_method))
        self.main()

    def main(self):
        if self.eval_method == "training":
            print("Calculating accuracy on training set...")

            features = Matrix(self.data, 0, 0, self.data.rows, self.data.cols-1)
            labels = Matrix(self.data, 0, self.data.cols-1, self.data.rows, 1)
            confusion = Matrix()
            start_time = time.time()
            self.learner.train(features, labels)
            elapsed_time = time.time() - start_time
            print("Time to train (in seconds): {}".format(elapsed_time))
            accuracy = self.learner.measure_accuracy(features, labels, confusion)
            print("Training set accuracy: " + str(accuracy))

            if self.print_confusion_matrix:
                print("\nConfusion matrix: (Row=target value, Col=predicted value)")
                confusion.print()
                print("")

        elif self.eval_method == "static":

            print("Calculating accuracy on separate test set...")

            test_data = Matrix(arff=self.eval_parameter)
            if self.normalize:
                test_data.normalize()

            print("Test set name: {}".format(self.eval_parameter))
            print("Number of test instances: {}".format(test_data.rows))
            features = Matrix(self.data, 0, 0, self.data.rows, self.data.cols-1)
            labels = Matrix(self.data, 0, self.data.cols-1, self.data.rows, 1)

            start_time = time.time()
            self.learner.train(features, labels)
            elapsed_time = time.time() - start_time
            print("Time to train (in seconds): {}".format(elapsed_time))

            train_accuracy = self.learner.measure_accuracy(features, labels)
            print("Training set accuracy: {}".format(train_accuracy))

            test_features = Matrix(test_data, 0, 0, test_data.rows, test_data.cols-1)
            test_labels = Matrix(test_data, 0, test_data.cols-1, test_data.rows, 1)
            confusion = Matrix()
            test_accuracy = self.learner.measure_accuracy(test_features, test_labels, confusion)
            print("Test set accuracy: {}".format(test_accuracy))

            if self.print_confusion_matrix:
                print("\nConfusion matrix: (Row=target value, Col=predicted value)")
                confusion.print()
                print("")

        elif self.eval_method == "random":
            """ This eval_method 1) creates a 'random' training/test split according to some user-specified percentage,
                            2) trains the data
                            3) reports training AND test accuracy
            """

            print("Calculating accuracy on a random hold-out set...")
            train_percent = float(self.eval_parameter)
            if train_percent < 0 or train_percent > 1:
                raise Exception("Percentage for random evaluation must be between 0 and 1")
            print("Percentage used for training: {}".format(train_percent))
            print("Percentage used for testing: {}".format(1 - train_percent))

            self.data.shuffle()

            train_size = int(train_percent * self.data.rows)
            train_features = Matrix(self.data, 0, 0, train_size, self.data.cols-1)
            train_labels = Matrix(self.data, 0, self.data.cols-1, train_size, 1)

            test_features = Matrix(self.data, train_size, 0, self.data.rows - train_size, self.data.cols-1)
            test_labels = Matrix(self.data, train_size, self.data.cols-1, self.data.rows - train_size, 1)

            start_time = time.time()
            self.learner.train(train_features, train_labels)
            elapsed_time = time.time() - start_time
            print("Time to train (in seconds): {}".format(elapsed_time))

            train_accuracy = self.learner.measure_accuracy(train_features, train_labels)
            print("Training set accuracy: {}".format(train_accuracy))

            confusion = Matrix()
            test_accuracy = self.learner.measure_accuracy(test_features, test_labels, confusion)
            print("Test set accuracy: {}".format(test_accuracy))

            if self.print_confusion_matrix:
                print("\nConfusion matrix: (Row=target value, Col=predicted value)")
                confusion.print()
                print("")

        elif self.eval_method == "cross":

            print("Calculating accuracy using cross-validation...")

            folds = int(self.eval_parameter)
            if folds <= 0:
                raise Exception("Number of folds must be greater than 0")
            print("Number of folds: {}".format(folds))
            reps = 1
            sum_accuracy = 0.0
            elapsed_time = 0.0
            for j in range(reps):
                self.data.shuffle()
                for i in range(folds):
                    begin = int(i * self.data.rows / folds)
                    end = int((i + 1) * self.data.rows / folds)

                    train_features = Matrix(self.data, 0, 0, begin, self.data.cols-1)
                    train_labels = Matrix(self.data, 0, self.data.cols-1, begin, 1)

                    test_features = Matrix(self.data, begin, 0, end - begin, self.data.cols-1)
                    test_labels = Matrix(self.data, begin, self.data.cols-1, end - begin, 1)

                    train_features.add(self.data, end, 0, self.data.cols - 1)
                    train_labels.add(self.data, end, self.data.cols - 1, 1)

                    start_time = time.time()
                    self.learner.train(train_features, train_labels)
                    elapsed_time += time.time() - start_time

                    accuracy = self.learner.measure_accuracy(test_features, test_labels)
                    sum_accuracy += accuracy
                    print("Rep={}, Fold={}, Accuracy={}".format(j, i, accuracy))

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

