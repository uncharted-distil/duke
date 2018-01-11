import sys
import json
import time
from datetime import datetime
from random import shuffle

import numpy as np

from identify_subject import getSentenceFromKeywords
from similarity_functions import (freq_nearest_similarity,
                                  get_class_similarities, w2v_similarity)
from utils import (NumpyEncoder, get_timestamp, load_dataset, load_model,
                   timeit, load_class_relationships)


def run_trial(data, classes, model, similarity_func=w2v_similarity, extra_args=None, max_num_samples=None):

    print('running trial with similarity function: {0}{1}\n'.format(similarity_func.__name__, ', and extra args: {0}'.format(extra_args) if extra_args else ''))
    
    similarities = {}
    for source, words in data.items():
        
        if max_num_samples and len(words) > max_num_samples:
            print('subsampling word list of length {0} to {1}'.format(len(words), max_num_samples))
            shuffle(words)
            words = words[:max_num_samples]  
        
        print('computing type similarity for ', source)
        similarities[source] = get_class_similarities(words, classes, model, similarity_func, extra_args)

    return similarities


def similarity_trials(
    model_name='wiki2vec',
    # dataset_name='LL0_1037_ada_prior',
    dataset_name='185_baseball',
    configs = [
        {'similarity_function': w2v_similarity},
        # {'similarity_function': freq_nearest_similarity},
        # {'similarity_function': freq_nearest_similarity, 'extra_args': {'n_nearest': 3}},
    ]):
    
    print('\n loading model')
    model = load_model(model_name)
    
    print('\n loading classes and relationship heirarchy')
    relationships = load_class_relationships(model)
    classes = relationships.keys()
    
    print('\n loading dataset')
    data = load_dataset(dataset_name, model)

    for conf in configs:
        sim_func = conf['similarity_function']
        extra_args = conf.get('extra_args')

        similarities = run_trial(data, classes, model, sim_func, extra_args)
        
        # write results of trial to file along with trial config info
        record = {
            'model': model_name, 
            'classes': classes, 
            'dataset': dataset_name,
            'sim_func': sim_func.__name__, 
            'extra_args': extra_args,
            'similarities': similarities,  # dict mapping column headers to similarity vectors (in the same order as classes list)
            }
        
        print('\n writing trial results to file')
        with open('trial-results/trial_{0}.json'.format(get_timestamp()), 'w') as f:
            json.dump(record, f, cls=NumpyEncoder, indent=4)

# def identify_subject(similarities):
    
    # top_n = {}
    # n = 10
    # parsedTypes = [''.join(a) for a in types]
    # for header in similarities.keys():
    #     best = sorted(zip(parsedTypes, similarities[header]), key=lambda x: x[1], reverse=True)[0:10]
    #     top_n[header] = best
    #     # print(header)
    #     # print(best)
    #     print(header + ": " + getSentenceFromKeywords(best))
    #     # print("===============")

    # all_types = []
    # for key in top_n.keys():
    #     all_types.extend(top_n[key])
    
    # print("Total: " + getSentenceFromKeywords(all_types))
    


if __name__ == '__main__':
    print(main(dataset_name=sys.argv[1]))
