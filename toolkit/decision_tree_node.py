import numpy as np

class TreeNode:

    decisions_made = []
    information = 0
    data_n = 0
    parentNode = None
    children = []
    feature_decided = None
    feature_value_decision = None
    count = 0
    data = np.zeros((1,1))

    def __init__(self):
        pass

    def add_decision(self, new_decision):
        self.decisions_made += [new_decision]
