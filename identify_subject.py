from __future__ import print_function
import sys
import json
import datetime
from PrettyPrinter import PrettyPrinter
from SentenceGenerator import SentenceGenerator
from KeywordSelector import KeywordSelector
from ScoreAccumulator import ScoreAccumulator

# This is the import to my custom handler -- can be replaced
# from nl import getTypeFromFile

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



def getSentenceFromKeywords(keywords, type_heirarchy_filename='type_heirarchy.json', verbose=False):
    # read in type heirarchy information
    with open(type_heirarchy_filename, 'r') as file:
        childParentMap = json.loads(file.read())
    with open('inverted_type_heirarchy.json', 'r') as file:
        parentChildMap = json.loads(file.read())

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


    scoreAccumulator = ScoreAccumulator(allTrees, parentChildMap, bestWordsMap)
    scores = scoreAccumulator.calculateScores(roots)
        
    # Print results prettily
    if(verbose):
        pp = PrettyPrinter(allTrees, scores, verbose=verbose)
        pp.prettyPrint(tree.keys(), "")
        print(pp.getString())
    # filename = 'results' + str(datetime.datetime.now()) + '.txt'
    # with open(filename, 'w+') as file:
    #     file.write(resultString)

    keywordSelector = KeywordSelector(tree, scores, childParentMap)
    keyword = keywordSelector.selectKeyword()

    sentenceGenerator = SentenceGenerator(keyword)
    sentence = sentenceGenerator.generate() 
    if(verbose):
        print(sentence)
    return sentence



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
