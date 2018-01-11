import sys
import json
import time
from datetime import datetime

import numpy as np

from identify_subject import getSentenceFromKeywords
from similarity_functions import (freq_nearest_similarity,
                                  get_type_similarities, w2v_similarity)
from utils import (NumpyEncoder, get_timestamp, load_dataset, load_model,
                   load_types, timeit)


def run_trial(data, types, model, similarity_func=w2v_similarity, extra_args=None):

    print('running trial with similarity function: {0}{1}\n'.format(similarity_func.__name__, ', and extra args: {0}'.format(extra_args) if extra_args else ''))
    
    similarities = {}
    for source, words in data.items():
        print('computing type similarity for ', source)
        similarities[source] = get_type_similarities(words, types, model, similarity_func, extra_args)

    return similarities


def main(
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
    
    print('\n loading types')
    types = load_types(model)
    
    print('\n loading dataset')
    data = load_dataset(dataset_name, model)

    for conf in configs:
        sim_func = conf['similarity_function']
        extra_args = conf.get('extra_args')

        similarities = run_trial(data, types, model, sim_func, extra_args)
        
        # write results of trial to file along with trial config info
        record = {
            'model': model_name, 
            'types': types, 
            'dataset': dataset_name,
            'sim_func': sim_func.__name__, 
            'extra_args': extra_args,
            'similarities': similarities,  # dict mapping column headers to similarity vectors (in the same order as types list)
            }
        
        print('\n writing trial results to file')
        with open('trial-results/trial_{0}.json'.format(get_timestamp()), 'w') as f:
            json.dump(record, f, cls=NumpyEncoder, indent=4)
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
