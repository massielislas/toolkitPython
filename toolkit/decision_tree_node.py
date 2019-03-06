import numpy as np

class TreeNode:

    decisions_made = []
    information = 0
    features_n = 0
    parent_node = None
    children = []
    feature_decided = None
    feature_value_decision = None
    features = None
    labels = None
    information_gain = 0


    def __init__(self):
        self.decisions_made = []
        self.information = 0
        self.features_n = 0
        self.parent_node = None
        self.children = []
        self.feature_decided = None
        self.feature_value_decision = None
        self.features = None
        self.labels = None
        pass

    def to_string(self):
        print('FEATURE DECIDED', self.feature_decided)
        print('FEATURE VALUE DECISION', self.feature_value_decision)
        print('DECISIONS MADE', self.decisions_made)

        print('LABELS')
        print(self.labels)

        print('FEATURES')
        print(self.labels)

        print('INFORMATION', self.information)
        print('INFORMATION GAIN', self.information_gain)

    def add_decision(self, new_decision):
        if self.parent_node is not None:
            if len(self.decisions_made) == 0:
                self.decisions_made += self.parent_node.decisions_made + [new_decision]
                self.feature_decided = new_decision

            else:
                self.decisions_made += [new_decision]
                self.feature_decided = new_decision
        else:
            print('PARENT IS NULL WHEN DECISION IS SET')

    def set_features(self, features):
        self.features = features
        self.features_n = len(features.data)
