import os
from toolkit import baseline_learner, utils, manager, matrix

# Full voting data, including missing values
# voting_data = "./datasets/voting.arff"
# voting_data_url = "http://axon.cs.byu.edu/data/uci_class/vote.arff"

## Download .arff data
iris_data = "./datasets/iris.arff"
iris_url = "http://axon.cs.byu.edu/data/uci_class/iris.arff"
utils.save_arff(iris_url, iris_data)

## Create a Matrix object from arff
iris = matrix.Matrix(arff=iris_data)

## Create manager
my_manager = manager.MLSystemManager()
args = r'-L baseline -A {} -E training'.format(iris_data)
my_manager.main(args)

print(my_manager.learner.average_label) # properties in learner
print(my_manager.data) # the Matrix class
print(my_manager.data.data) # the numpy array of the matrix class