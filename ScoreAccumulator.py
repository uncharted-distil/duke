'''
Calculate a score for every node, where the value is the sum of its value
and the value of all of its children
allTrees: a map between each type and any sub-types it has, including only
    those types found in this dataset
parentChildMap: a map between every possible type and its subtypes
wordmap: a map between each identified type and its score
'''
class ScoreAccumulator:
    def __init__(self, allTrees, parentChildMap, wordmap):
        self.allTrees = allTrees
        self.parentChildMap = parentChildMap
        self.initialWordmap = wordmap
        self.scores = {}

    def calculateScores(self, roots):
        for root in roots:
            self.scoreNode(root)
        return self.scores

    def scoreNode(self, root_node):
        self.scores[str(root_node)] = self.initialWordmap.get(root_node, 0)
        if root_node in self.allTrees.keys():
            for node in self.allTrees[root_node]:
                if not node in self.scores.keys():
                    self.scoreNode(node)
            # METRIC
            # Accumulation function
            childScores = [self.scores[str(node)] for node in self.allTrees[root_node]]

            # sum of children
            # self.scores[str(root_node)] = sum(childScores)

            # max of children
            # self.scores[srt(root_node)] = max(childScores)

            # average of children:
            # self.scores[str(root_node)] = sum(childScores) / float(len(childScores))

            # sum of children + self
            # self.scores[str(root_node)] = sum(childScores) + self.scores[str(root_node)]

            # sum of children + self / total possible children of node
            if root_node in self.parentChildMap.keys():
                weight = (float(len(childScores)) / float(len(self.parentChildMap[root_node])))
            else:
                weight = 1.0
            self.scores[str(root_node)] = (sum(childScores) + self.scores[str(root_node)]) * weight


            # max of children + self:
            # self.scores[srt(root_node)] = max(childScores) + self.scores[str(root_node)]

            # average of children + self
            # self.scores[str(root_node)] = (sum(childScores) + self.scores[str(root_node)]) / float(len(childScores) + 1)
        return self.scores
