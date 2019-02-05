



## Download .arff data
iris_data = "./datasets/iris.arff"
iris_url = "http://axon.cs.byu.edu/data/uci_class/iris.arff"
utils.save_arff(iris_url, iris_data)

## Create manager - from commandline argument
if False:
    args = r'-L baseline -A {} -E training'.format(iris_data)
    my_manager = manager.MLSystemManager()
    session = my_manager.create_session_from_argv(args)

    print(session.learner.average_label) # properties in learner
    print(session.data) # the Matrix class
    print(session.data.data) # the numpy array of the matrix class
