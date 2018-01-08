from __future__ import print_function
import sys
import json
from inflector import English

# This is the import to my custom handler -- can be replaced
from nl import getTypeFromFile

# Build dictionary containing the heirarchy of types. If a type has sub-types
# in the results, then the value at htat type will be another dictionary, but if
# the type is a leaf, then the value will be None
def getLeaves(allTrees, node):
    results = {}
    for leaf in allTrees[node]:
        if leaf in allTrees.keys():
            results[leaf] = getLeaves(allTrees, leaf)
        else:
            results[leaf] = None
    return results

# Calculate a score for every node, where the value is the sum of its value
# and the value of all of its children
def scoreNode(allTrees, leafScores, allScores, root_node):
    allScores[str(root_node)] = leafScores.get(root_node, 0)
    if root_node in allTrees.keys():
        for node in allTrees[root_node]:
            if not node in allScores.keys():
                allScores = scoreNode(allTrees, leafScores, allScores, node)
            # Accumulation function
            allScores[str(root_node)] = allScores[str(root_node)] + allScores[str(node)]
    return allScores

# Create a string which displays in a 'pretty' fashion the tree of types and 
# the values of each type, sorted in descending order
def prettyPrint(tree, scores, prefix, resultString, verbose=False):
    dicts = filter(lambda x: not tree[x] is None, tree.keys())
    leafs = filter(lambda x: tree[x] is None, tree.keys())

    keyScores = [(key, scores[key]) for key in dicts]
    keyScores.sort(key=lambda x: x[1], reverse=True)
    for keyScore in keyScores:
        currString = prefix + keyScore[0] + "(" + str(keyScore[1]) + ")"
        if(verbose):
            print(currString)
        resultString += currString
        prettyPrint(tree[keyScore[0]], scores, prefix + "\t", resultString, verbose)

    keyScores = [(key, scores[key]) for key in leafs]
    keyScores.sort(key=lambda x: x[1], reverse=True)
    for keyScore in keyScores:
        currString = prefix + keyScore[0] + "(" + str(keyScore[1]) + ")"
        if(verbose):
            print(currString)
        resultString += currString
    return resultString
               

# Create dictionary of all parent types and their children
def buildCompleteTypeSet(bestWordsMap, childParentMap):
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


# get best terms from model
# NOTE: These function calls can be replaced by any calls that simply return a list of tuples
# where the first value is the type string, and the second value is a number representing its 
# likelihood. Right now, I'm assuming it is a count, but other metrics will work as well.
columns = getTypeFromFile(sys.argv[1], sys.argv[2], HEADER=True)
words = getTypeFromFile(sys.argv[1], sys.argv[2])

# read in type heirarchy information
with open('type_heirarchy_parents.json', 'r') as file:
    childParentMap = json.loads(file.read())

# Filter out best words from model
# NOTE: Hopefully this will not be needed once model is improved
maxCountColumns = columns[0][1]
maxCountWords = words[0][1]

topTenPercent = int(float(maxCountWords) * 0.5)
bestWordsMap = {a[0]: a[1] for a in filter(lambda x: x[1] > 1, words)}


# Create dictionary of all parent types in this dataset, as well
# as all of the relevant children
allTrees = buildCompleteTypeSet(bestWordsMap, childParentMap)

# 
nodes = allTrees.keys()
roots = set()
leaves = set()

for treeKey in allTrees.keys():
    for leaf in allTrees[treeKey]:
        leaves.add(leaf)

for node in nodes:
    if not node in leaves:
        roots.add(node)

# Create tree representing all types in dataset
tree = {}
for root in roots:
    tree[str(root)] = getLeaves(allTrees, root) 

# Create map from each type to its accumulated value
scores = {}
for root in roots:
    scores.update(scoreNode(allTrees, bestWordsMap, {}, root))
    
# Print results prettily
resultString = prettyPrint(tree, scores, "", "", verbose=True)
with open('results.txt', 'w') as file:
    file.write(resultString)


# Largely ignore all work done above and print results from original model

inf = English()

print("This dataset is about ", end = '')
addAnd = False
for word in columns:
    if word[1] == maxCountColumns:
        if not addAnd:
            addAnd = True
            print(inf.pluralize(word[0]), end='')
        else:
            print(" and " + inf.pluralize(word[0]), end = '')
print("")

addAnd = False
print("The column " + sys.argv[2] + " is about ", end ='')
for word in words:
    if word[1] == maxCountWords:
        if not addAnd:
            addAnd = True
            print(inf.pluralize(word[0]), end ='')
        else:
            print(" and " + inf.pluralize(word[0]), end='')
            
print("")