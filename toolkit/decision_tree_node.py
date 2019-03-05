import numpy as np

class TreeNode:

    decisions_made = []
    information = 0
    data_n = 0
    parentNode = None
    children = []
    feature_decided = None
    feature_value_decision = None
    feature_value_decision_count = 0
    data = np.zeros((1,1))

    def __init__(self):
        pass

    def add_decision(self, new_decision):
        self.decisions_made += [new_decision]
        self.feature_decided = new_decision

    def set_data(self, data):
        self.data = data
        self.data_n = len(data)
