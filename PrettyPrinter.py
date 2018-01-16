# Create a string which displays in a 'pretty' fashion the tree of types and 
# the values of each type, sorted in descending order
class PrettyPrinter:
    def __init__(self, allTrees, scores, verbose=False):
        self.allTrees = allTrees
        self.scores = scores
        self.resultString = ""
        self.verbose = verbose
        self.parents = set(filter(lambda x: not allTrees[x] is None, allTrees.keys()))
        self.leafs = set(filter(lambda x: allTrees[x] is None, allTrees.keys()))

    def prettyPrint(self, keywords, prefix):
        parentKeyScores = [(keyword, self.scores.get(keyword, 0)) for keyword in keywords if keyword in self.parents]
        parentKeyScores.sort(key=lambda x: x[1], reverse=True)

        leafKeyScores = [(keyword, self.scores.get(keyword, 0)) for keyword in keywords if keyword in self.leafs]
        leafKeyScores.sort(key=lambda x: x[1], reverse=True)

        for keyScore in parentKeyScores:
            currString = prefix + keyScore[0] + "(" + str(keyScore[1]) + ")"
            if(self.verbose):
                print(currString)
            self.resultString += currString + "\n"
            self.prettyPrint(self.allTrees[keyScore[0]], prefix + "\t")

        for keyScore in leafKeyScores:
            currString = prefix + keyScore[0] + "(" + str(keyScore[1]) + ")"
            if(self.verbose):
                print(currString)
            self.resultString += currString + "\n"

    def getString(self):
        return self.resultString