
class TreeBuilder:
    # Build dictionary containing the heirarchy of types. If a type has sub-types
    # in the results, then the value at htat type will be another dictionary, but if
    # the type is a leaf, then the value will be None
    def buildCompleteTree(self, allTrees, roots):
        tree = {}
        for root in roots:
            tree[str(root)] = self.getLeafs(allTrees, root) 
        return tree

    def getLeafs(self, allTrees, node):
        results = {}
        for leaf in allTrees[node]:
            if leaf in allTrees.keys():
                results[leaf] = self.getLeafs(allTrees, leaf)
            else:
                results[leaf] = None
        return results

    # Create dictionary of all parent types and their children
    def buildAllSubTrees(self, bestWordsMap, childParentMap): 
        allTrees = dict()
        completeWords = set()
        for word in bestWordsMap.keys():
            completeWords.add(word)

        updatesFlag = True

        while updatesFlag:
            updatesFlag = False
            updates = set()
            for word in completeWords:
                if word in childParentMap.keys():
                    parents = childParentMap[word]
                    for parent in parents:
                        if not parent in completeWords:
                            updatesFlag = True
                            updates.add(parent)
            completeWords = completeWords.union(updates)

        for word in completeWords:
            if word in childParentMap.keys():
                for parent in childParentMap[word]:
                    if parent in allTrees.keys():
                        allTrees[str(parent)].append(word)
                    else:
                        allTrees[str(parent)] = [word]
        return allTrees
