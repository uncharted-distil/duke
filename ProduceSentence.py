import json
import time
import pandas as pd
from datetime import datetime

import numpy as np

from identify_subject import getSentenceFromKeywords
from similarity_functions import (freq_nearest_similarity,
                                  get_type_similarities, w2v_similarity)
from utils import (get_timestamp, load_dataset, load_model, load_types,
                   log_top_similarities, timeit, NumpyEncoder)

class ProduceSentence():
    def __init__(self,
    model_name='/data/duke/models/word2vec/en_1000_no_stem/en.model',
    types_filename='/data/duke/models/ontologies/types',
    type_heirarchy_filename='type_heirarchy.json',
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
        self.types = load_types(self.model, types_filename=types_filename)

        self.configs = configs
        self.type_heirarchy_filename = type_heirarchy_filename

    def run_trial(self, data, similarity_func=w2v_similarity, extra_args=None):

        if(self.verbose):
            print('running trial with similarity function: {0}{1}\n'.format(similarity_func.__name__, ', and extra args: {0}'.format(extra_args) if  extra_args else ''))
        
        similarities = {}
        for source, words in data.items():
            if(self.verbose):
                print('computing type similarity for ', source)
            similarities[source] = get_type_similarities(words, self.types, self.model, similarity_func, extra_args)

        return similarities
    
    def produceSentenceFromDataframe(self, full_df):
        if(self.verbose):
            print('loading dataset')
        data = load_dataset(full_df, self.model, verbose=self.verbose)

        results = {}
        for conf in self.configs:
            sim_func = conf['similarity_function']
            extra_args = conf.get('extra_args')

            similarities = self.run_trial(data, sim_func, extra_args)
            
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

        top_n = {}
        n = 10
        returnSentence = ""
        parsedTypes = [''.join(a) for a in self.types]
        for header in similarities.keys():
            best = sorted(zip(parsedTypes, similarities[header]), key=lambda x: x[1], reverse=True)[0:n]
            top_n[header] = best
            returnSentence = returnSentence + header + ": " + getSentenceFromKeywords(best, type_heirarchy_filename=self.type_heirarchy_filename, verbose=False) + "\n" 
            if(self.verbose):
                print(best)
                print(header + ": " + getSentenceFromKeywords(best, type_heirarchy_filename=self.type_heirarchy_filename, verbose=True))

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
        
        returnSentence = returnSentence + "\nOverll data contents: " + getSentenceFromKeywords(all_types, type_heirarchy_filename=self.type_heirarchy_filename, verbose=self.verbose)

        return returnSentence