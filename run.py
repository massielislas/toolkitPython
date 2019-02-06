from toolkit import baseline_learner, manager, arff
import numpy as np
from toolkit import supervised_learner, manager, arff

arff_path = r"./test/datasets/creditapproval.arff"

# Session can take either instantiated or uninstantiated learner
my_learner = baseline_learner.BaselineLearner
credit_approval = arff.Arff(arff=arff_path)

# Manual Training/Test
session = manager.ToolkitSession(arff=credit_approval, learner=my_learner)
train_features, train_labels, test_features, test_labels = session.training_test_split(.7)  # 70% training
session.train(train_features, train_labels)
session.test(test_features, test_labels)
print(session.training_accuracy)

# Pass on hyperparameters to learner
session = manager.ToolkitSession(arff=credit_approval, learner=my_learner, data=credit_approval, example_hyperparameter=.5)
print(session.learner.data_shape, (690, 16))
print(session.learner.example_hyperparameter, .5)

# Automatic
session2 = manager.ToolkitSession(arff=credit_approval, learner=my_learner, eval_method="random", eval_parameter=.7)

# Cross validate
session3 = manager.ToolkitSession(arff=credit_approval, learner=my_learner)
session3.cross_validate(folds=10, reps=3)
print(session3.test_accuracy)

