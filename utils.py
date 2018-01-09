import re
import time

import numpy as np
import pandas as pd
from gensim.models import Word2Vec

def timeit(func, args=None):
    start = time.time()
    print('calling {0} \n'.format(func.__name__))
    result = func(*args) if args else func()
    print('{0} took {1} seconds \n\n'.format(func.__name__, time.time() - start))
    return result


def load_model(model_name='wiki2vec'):
    ''' Load a word2vec model from a file in models/ '''
    models = {
        'wiki2vec': 'en_1000_no_stem/en.model',  # file paths hard coded here
    }
    start = time.time()
    model = Word2Vec.load("models/word2vec/{0}".format(models[model_name]))
    return model


def load_dataset(dataset_name, model):
    csv_path = 'data/{0}/{0}_dataset/tables/learningData.csv'.format(dataset_name)
    full_df = pd.read_csv(csv_path, header=0)  # read csv assuming first line has header text
    text_df = full_df.select_dtypes(['object'])  # drop non-text rows (pandas strings are of type 'object')
    # TODO confirm that the columns selected can't be cast to a numeric type to avoid numeric strings (e.g. '1')
    
    # concat then normalize headers and text columns
    headers = full_df.columns.values
    headers = normalize_headers(headers, model)
    
    text = [normalize_text(text_df[h].values, model) for h in text_df.columns]
    text = np.concatenate(text)  # flatten columns into single vector of word lists
    
    return headers, text  


def load_types(model):
    # load types and normalize (remove out of vocab etc.)
    with open('models/ontologies/types', 'r') as f:  
        types = f.read().splitlines()
        return normalize_types(types, model)  


def normalize_types(types, model):
    # create a lol of types split by capitalization
    types = np.array([re.findall('[A-Z][^A-Z]*', typ) for typ in types])  # list of lists of single words
    # TODO more general processing? split by spaces?
    # remove types with out-of-vocab words

    in_vocab = [np.all([word in model.wv.vocab for word in typ]) for typ in types]   
    return types[in_vocab]


def normalize_headers(headers, model):
    headers = np.array([h.replace('_', ' ').replace('-', ' ').lower().split(' ') for h in headers])  # list of lists of single words
    in_vocab = [np.all([word in model.wv.vocab for word in h]) for h in headers]   
    return headers[in_vocab]


def normalize_text(text, model):
    result = []
    for t in text:
        if (t is not '') and isinstance(t, str):
            result.append(t.replace('_', ' ').replace('-', ' ').lower().split(' '))
        else: 
            print('invalid text: ', t)
    in_vocab = [np.all([word in model.wv.vocab for word in t]) for t in result]
    return np.array(result)[in_vocab] 


def log_top_similarities(sorted_types, similarities, n_keep=20):
    print('top {0} types, similarities: \n'.format(n_keep))
    for typ, sim in zip(sorted_types[:n_keep], similarities[:n_keep]):
        print(typ, sim)
    print('\n\n')