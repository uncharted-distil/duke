'''
Given an arbitrary type tree, a set of corresponding scores, and a map from child
nodes to their parents, select the best keyword from the tree
'''
class KeywordSelector:
    def __init__(self, tree, scores, childParentMap):
        self.tree = tree
        self.scores = scores
        self.childParentMap = childParentMap
        self.selectedKeyword = ""

    def selectKeyword(self):
        self.maxLeaf = self.findMaxLeaf(self.tree)[0]
        self.selectedKeyword = self.getDominantAncestor(self.maxLeaf)
        return self.selectedKeyword

    def getSelectedKeyword(self):
        if(self.selectedKeyword == ""):
            return self.selectKeyword()
        else:
            return self.selectedKeyword

    def findMaxLeaf(self, tree):
        keyScores = [(key, self.scores[key]) for key in tree.keys()]
        maxKeyScore = max(keyScores, key=lambda x: x[1])
        if(tree[maxKeyScore[0]] is None):
            return maxKeyScore
        else:
            return self.findMaxLeaf(tree[maxKeyScore[0]])

    # Given a leaf node, walk up the tree until the 'dominant ancestor' is found
    # this dominant ancestor is determined by the if statement below
    def getDominantAncestor(self, node):
        if not node in self.childParentMap.keys():
            return node
        parents = self.childParentMap[node]
        parent = max([(p, self.scores[p]) for p in parents], key=lambda x: x[1])[0]
        nodeScore = self.scores[node]
        parentScore = self.scores[parent]
        # METRIC
        # Metric chosen largely at random and with minimal justification
        if(parentScore > nodeScore):
            return self.getDominantAncestor(parent)
        else:
            return node