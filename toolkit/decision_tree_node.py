import numpy as np

class TreeNode:

    decisions_made = []
    information = 0
    data_n = 0
    parent_node = None
    children = []
    feature_decided = None
    feature_value_decision = None
    data = np.zeros((1,1))
    labels = np.zeros((1,1))
    information_gain = 0


    def __init__(self):
        self.decisions_made = []
        self.information = 0
        self.data_n = 0
        self.parent_node = None
        self.children = []
        self.feature_decided = None
        self.feature_value_decision = None
        self.data = np.zeros((1,1))
        self.labels = np.zeros((1,1))
        pass

    def add_decision(self, new_decision):
        self.decisions_made += [new_decision]
        self.feature_decided = new_decision

    def set_data(self, data):
        self.data = data
        self.data_n = len(data)
