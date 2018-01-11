from __future__ import print_function
import sys
import json
import datetime
from inflector import English

# This is the import to my custom handler -- can be replaced
# from nl import getTypeFromFile

# Build dictionary containing the heirarchy of classes. If a class has sub-classes
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
        # METRIC
        # Accumulation function
        childScores = [allScores[str(node)] for node in allTrees[root_node]]

        # max of children
        # allScores[srt(root_node)] = max(childScores)

        # average of children:
        # allScores[str(root_node)] = sum(childScores) / float(len(childScores))

        # max of children + self:
        # allScores[srt(root_node)] = max(childScores) + allScores[str(root_node)]

        # average of children + self
        allScores[str(root_node)] = (sum(childScores) + allScores[str(root_node)]) / float(len(childScores) + 1)
    return allScores

# Create a string which displays in a 'pretty' fashion the tree of classes and 
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
        resultString += currString + "\n"
        resultString += prettyPrint(tree[keyScore[0]], scores, prefix + "\t", resultString, verbose)

    keyScores = [(key, scores[key]) for key in leafs]
    keyScores.sort(key=lambda x: x[1], reverse=True)
    for keyScore in keyScores:
        currString = prefix + keyScore[0] + "(" + str(keyScore[1]) + ")"
        if(verbose):
            print(currString)
        resultString += currString + "\n"
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

def getMaxLeaf(tree, scores):
    keyScores = [(key, scores[key]) for key in tree.keys()]
    maxKeyScore = max(keyScores, key=lambda x: x[1])
    if(tree[maxKeyScore[0]] is None):
        return maxKeyScore
    else:
        return getMaxLeaf(tree[maxKeyScore[0]], scores)

# Given a leaf node, walk up the tree until the 'dominant ancestor' is found
# this dominant ancestor is determined by the if statement below
def getDominantAncestor(childParentMap, scores, node):
    parents = childParentMap[node]
    parent = max([(p, scores[p]) for p in parents], key=lambda x: x[1])[0]
    nodeScore = scores[node]
    parentScore = scores[parent]
    # METRIC
    # Metric chosen largely at random and with minimal justification
    if(parentScore > nodeScore):
        return getDominantAncestor(childParentMap, scores, parent)
    else:
        return node

def getSentenceFromKeywords(keywords, verbose=False):
    # read in type heirarchy information
    with open('type_heirarchy.json', 'r') as file:
        childParentMap = json.loads(file.read())
    # with open('inverted_type_heirarchy.json', 'r') as file:
    #     parentChildMap = json.loads(file.read())

    # Filter out best words from model
    # NOTE: Hopefully this will not be needed once model is improved
    # bestWordsMap = {a[0]: a[1] for a in filter(lambda x: x[1] > 1, keywords)}
    bestWordsMap = {a[0]: a[1] for a in keywords}


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
    resultString = prettyPrint(tree, scores, "", "", verbose=verbose)
    filename = 'results' + str(datetime.datetime.now()) + '.txt'
    with open(filename, 'w+') as file:
        file.write(resultString)



    # Largely ignore all work done above and print results from original model

    inf = English()

    sentence_prefix = "This dataset is about "
    leafTerm = getMaxLeaf(tree, scores)
    word = getDominantAncestor(childParentMap,scores,leafTerm[0])
    sentence = sentence_prefix + inf.pluralize(word)
    if(verbose):
        print(sentence)
    return(sentence)

# Sample for how to use getSentenceFromKeywords
def main(args):
    # get best terms from model
    # NOTE: These function calls can be replaced by any calls that simply return a list of tuples
    # where the first value is the type string, and the second value is a number representing its 
    # likelihood. Right now, I'm assuming it is a count, but other metrics will work as well.
    # words = getTypeFromFile(args[1], args[2])
    # sentence = getSentenceFromKeywords(words, verbose=True)
    pass
    

if __name__ == '__main__':
    main(sys.argv)
