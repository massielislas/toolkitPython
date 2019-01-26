from __future__ import (absolute_import, division, print_function, unicode_literals)

from .supervised_learner import SupervisedLearner
from .baseline_learner import BaselineLearner
from .matrix import Matrix
import random
import argparse
import time
import sys
import textwrap
import shlex

class MLSystemManager:
    def __init__(self, args=None, learner=None):
        self.learner = learner
        self.data = None
        self.doc_string = textwrap.dedent('''
                            ML toolkit manager
                            Usage: python toolkit.manager -L [learningAlgorithm] -A [ARFF_File] -E [EvaluationMethod] {[ExtraParamters]} [-N] [-R seed]
                            
                            Possible evaluation methods are:
                            python toolkit.manager -L [learningAlgorithm] -A [ARFF_File] -E training
                            python toolkit.manager -L [learningAlgorithm] -A [ARFF_File] -E static [TestARFF_File]
                            python toolkit.manager -L [learningAlgorithm] -A [ARFF_File] -E random [PercentageForTraining]
                            python toolkit.manager -L [learningAlgorithm] -A [ARFF_File] -E cross [numOfFolds]''')
        self.main(args)

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
            "baseline": BaselineLearner(),
            #"perceptron": PerceptronLearner(),
            #"neuralnet": NeuralNetLearner(),
            #"decisiontree": DecisionTreeLearner(),
            #"knn": InstanceBasedLearner()
        }
        if model in modelmap:
            return modelmap[model]
        else:
            raise Exception("Unrecognized model: {}".format(model))

    def main(self, myArgs=None):
        # parse the command-line arguments
        if myArgs is None:
            args = self.parser().parse_args()
        else:
            # user can pass in command line argument as a string from another Python script
            args = self.parser().parse_args(shlex.split(myArgs))

        file_name = args.arff
        learner_name = args.L
        eval_method = args.E[0]
        eval_parameter = args.E[1] if len(args.E) > 1 else None
        print_confusion_matrix = args.verbose
        normalize = args.normalize
        random.seed(args.seed) # Use a seed for deterministic results, if provided (makes debugging easier)

        # load the model
        if self.learner is None:
            self.learner = self.get_learner(learner_name)

        # load the ARFF file
        self.data = Matrix()
        self.data.load_arff(file_name)
        if normalize:
            print("Using normalized data")
            self.data.normalize()

        # print some stats
        print("\nDataset name: {}\n"
              "Number of instances: {}\n"
              "Number of attributes: {}\n"
              "Learning algorithm: {}\n"
              "Evaluation method: {}\n".format(file_name, self.data.rows, self.data.cols, learner_name, eval_method))

        if eval_method == "training":

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

            if print_confusion_matrix:
                print("\nConfusion matrix: (Row=target value, Col=predicted value)")
                confusion.print()
                print("")

        elif eval_method == "static":

            print("Calculating accuracy on separate test set...")

            test_data = Matrix(arff=eval_parameter)
            if normalize:
                test_data.normalize()

            print("Test set name: {}".format(eval_parameter))
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

            if print_confusion_matrix:
                print("\nConfusion matrix: (Row=target value, Col=predicted value)")
                confusion.print()
                print("")

        elif eval_method == "random":
            """ This eval_method 1) creates a 'random' training/test split according to some user-specified percentage,
                            2) trains the data
                            3) reports training AND test accuracy
            """

            print("Calculating accuracy on a random hold-out set...")
            train_percent = float(eval_parameter)
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

            if print_confusion_matrix:
                print("\nConfusion matrix: (Row=target value, Col=predicted value)")
                confusion.print()
                print("")

        elif eval_method == "cross":

            print("Calculating accuracy using cross-validation...")

            folds = int(eval_parameter)
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
            raise Exception("Unrecognized evaluation method '{}'".format(eval_method))

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

class ToolkitArgParser(argparse.ArgumentParser):
    def __init__(self, description=None, doc_string=None):
        super(ToolkitArgParser, self).__init__(description=description)
        self.doc_string = doc_string

    def error(self, message):
        sys.stderr.write('Error: %s\n' % message)
        print(self.doc_string)
        sys.exit()


if __name__ == '__main__':
    MLSystemManager().main()
