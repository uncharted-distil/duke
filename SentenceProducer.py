import json
import time
import timeit
import pandas as pd
from datetime import datetime

import numpy as np
from TreeBuilder import TreeBuilder
from PrettyPrinter import PrettyPrinter
from KeywordSelector import KeywordSelector
from ScoreAccumulator import ScoreAccumulator
from SentenceGenerator import SentenceGenerator

from similarity_functions import (freq_nearest_similarity,
                                  get_class_similarities, w2v_similarity)
from utils import (get_timestamp, load_dataset, load_model, load_ontology,
                   print_top_similarities, timeit, NumpyEncoder)

class SentenceProducer:
    def __init__(self,
    model_name='/data/duke/models/word2vec/en_1000_no_stem/en.model',
    types_filename='/data/duke/models/ontologies/types',
    type_hierarchy_filename='type_hierarchy.json',
    inverted_type_hierarchy_filename='inverted_type_hierarchy.json',
    configs = [
        {'similarity_function': w2v_similarity},
        # {'similarity_function': freq_nearest_similarity},
        # {'similarity_function': freq_nearest_similarity, 'extra_args': {'n_nearest': 3}},
    ],
    verbose=False):
        self.verbose = verbose
        if(self.verbose):
            print('loading model')
        self.model_name = model_name
        self.model = load_model(model_name)
        
        if(self.verbose):
            print('loading types')
        self.types_filename = types_filename
        self.types = load_ontology(self.model, types_filename=types_filename)

        self.configs = configs
        self.type_hierarchy_filename = type_hierarchy_filename
        self.inverted_type_hierarchy_filename = inverted_type_hierarchy_filename

    def run_trial(self, data, similarity_func=w2v_similarity, extra_args=None):

        if(self.verbose):
            print('running trial with similarity function: {0}{1}\n'.format(similarity_func.__name__, ', and extra args: {0}'.format(extra_args) if  extra_args else ''))
        
        similarities = {}
        for source, words in data.items():
            if(self.verbose):
                print('computing type similarity for ', source)
            similarities[source] = get_class_similarities(words, self.types, self.model, similarity_func, extra_args)

        return similarities

    def calculateSimilarities(self, data):
        results = {}
        similaritiesList = []
        for conf in self.configs:
            sim_func = conf['similarity_function']
            extra_args = conf.get('extra_args')

            similarities = self.run_trial(data, sim_func, extra_args)
            similaritiesList.append(similarities)
            
            # write results of trial to file along with trial config info
            record = {
                'model': self.model_name, 
                'types': self.types, 
                'sim_func': sim_func.__name__, 
                'extra_args': extra_args,
                'similarities': similarities,  # dict mapping column headers to similarity vectors (in the same order as types list)
                }
            
            with open('trials/trial{0}'.format(get_timestamp()), 'w') as f:
                json.dump(record, f, cls=NumpyEncoder)
        return similaritiesList 

    def getSentenceFromKeywords(self, keywords, type_hierarchy_filename='type_hierarchy.json'):
        # read in type hierarchy information
        with open(self.type_hierarchy_filename, 'r') as file:
            childParentMap = json.loads(file.read())
        with open(self.inverted_type_hierarchy_filename, 'r') as file:
            parentChildMap = json.loads(file.read())

        # Filter out best words from model
        # NOTE: Hopefully this will not be needed once model is improved
        # bestWordsMap = {a[0]: a[1] for a in filter(lambda x: x[1] > 1, keywords)}
        bestWordsMap = {a[0]: a[1] for a in keywords}


        # Create dictionary of all parent types in this dataset, as well
        # as all of the relevant children
        treeBuilder = TreeBuilder()
        allTrees = treeBuilder.buildAllSubTrees(bestWordsMap, childParentMap)

        # build set of types that do not have any parents
        leafs = set([leaf for key in allTrees.keys() for leaf in allTrees[key]])
        roots = allTrees.keys() - leafs

        scoreAccumulator = ScoreAccumulator(allTrees, parentChildMap, bestWordsMap)
        scores = scoreAccumulator.calculateScores(roots)
            
        # Create tree representing all types in dataset
        tree = treeBuilder.buildCompleteTree(allTrees, roots)

        # Print results prettily
        if(self.verbose):
            pp = PrettyPrinter(allTrees, scores, verbose=self.verbose)
            pp.prettyPrint(tree.keys(), "")
            print(pp.getString())
        # filename = 'results' + str(datetime.datetime.now()) + '.txt'
        # with open(filename, 'w+') as file:
        #     file.write(resultString)

        keywordSelector = KeywordSelector(tree, scores, childParentMap)
        keyword = keywordSelector.selectKeyword()

        sentenceGenerator = SentenceGenerator(keyword)
        sentence = sentenceGenerator.generate() 
        if(self.verbose):
            print(sentence)
        return sentence

    
    def produceSentenceFromDataframe(self, full_df):
        if(self.verbose):
            print('loading dataset')
        data = load_dataset(full_df, self.model, verbose=self.verbose)


        simStart = time.time()
        similaritiesList = self.calculateSimilarities(data)
        
        treeStart = time.time()
        # naively just pull off first simliarities dict
        similarities = similaritiesList[0]

        top_n = {}
        n = 10
        returnSentence = ""
        parsedTypes = [''.join(a) for a in self.types]
        for header in similarities.keys():
            best = sorted(zip(parsedTypes, similarities[header]), key=lambda x: x[1], reverse=True)[0:n]
            top_n[header] = best
            returnSentence = returnSentence + header + ": " + self.getSentenceFromKeywords(best, type_hierarchy_filename=self.type_hierarchy_filename) + "\n" 
            if(self.verbose):
                print(best)
                print(header + ": " + self.getSentenceFromKeywords(best, type_hierarchy_filename=self.type_hierarchy_filename))

        all_types = []
        for key in top_n.keys():
            all_types.extend(top_n[key])
        all_types_aggregated = {}
        for t_score in all_types:
            all_types_aggregated[t_score[0]] = 0.0
        for t_score in all_types:
            all_types_aggregated[t_score[0]] = all_types_aggregated[t_score[0]] + t_score[1] 

        all_types = []
        for key in all_types_aggregated.keys():
            all_types.append((key, all_types_aggregated[key]))
        
        returnSentence = returnSentence + "\nOverll data contents: " + self.getSentenceFromKeywords(all_types, type_hierarchy_filename=self.type_hierarchy_filename)
        return returnSentence