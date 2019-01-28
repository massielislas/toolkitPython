import os
from toolkit import baseline_learner, utils, manager, matrix

""" This is an example script of how you might automate calls to the MLToolkit. 
If you a running Python interactively (e.g. in terminal, IPython, Jupyter, etc.), this may useful as you can access predictions/weights after training has taken place.
"""


## From commandline (example):
# python -m toolkit.manager -L baseline -A ./datasets/iris.arff -E random .7

# Full voting data, _including_ missing values
# voting_data = "./datasets/voting.arff"
# voting_data_url = "http://axon.cs.byu.edu/data/uci_class/vote.arff"

## Download .arff data
iris_data = "./datasets/iris.arff"
iris_url = "http://axon.cs.byu.edu/data/uci_class/iris.arff"
utils.save_arff(iris_url, iris_data)

## Create manager - from commandline argument
args = r'-L baseline -A {} -E training'.format(iris_data)
my_manager = manager.MLSystemManager()
session = my_manager.create_session_from_argv(args)

print(session.learner.average_label) # properties in learner
print(session.data) # the Matrix class
print(session.data.data) # the numpy array of the matrix class

## Create manager -- from another Python Script with custom learner
my_learner = baseline_learner.BaselineLearner
session = my_manager.create_new_session(arff_file=iris_data, learner=my_learner, eval_method="training", eval_parameter=None, print_confusion_matrix=False, normalize=False, random_seed=None)

## Create a Matrix object from arff
iris = matrix.Matrix(arff=iris_data)
