import re
import time
from datetime import datetime
import json
from inflection import underscore

import numpy as np
import pandas as pd
from gensim.models import Word2Vec

from ontologies.ontology import get_relationships_file_name



def timeit(func, args=None):
    start = time.time()
    print('calling {0} \n'.format(func.__name__))
    result = func(*args) if args else func()
    print('{0} took {1} seconds \n\n'.format(func.__name__, time.time() - start))
    return result


def load_model(model_name='wiki2vec'):
    ''' Load a word2vec model from a file in models/ '''
    models = {
        'wiki2vec': 'en_1000_no_stem/en.model',  # w2v model file paths hard coded here
    }
    return Word2Vec.load("/data/duke/models/word2vec/{0}".format(models[model_name]))


def get_dropped(all_headers, new_headers):
    return set(all_headers).difference(set(new_headers))


def load_dataset(dataset_name, model, drop_nan=True):
    csv_path = '/data/duke/data/{0}/{0}_dataset/tables/learningData.csv'.format(dataset_name)
    full_df = pd.read_csv(csv_path, header=0)  # read csv assuming first line has header text. TODO handle files w/o headers
    headers = full_df.columns.values

    # TODO confirm that the columns selected can't be cast to a numeric type to avoid numeric strings (e.g. '1')
    text_df = full_df.select_dtypes(['object'])  # drop non-text rows (pandas strings are of type 'object')
    dtype_dropped = get_dropped(headers, text_df.columns.values)
    print('dropped non-text columns: {0} \n'.format(list(dtype_dropped)))

    if drop_nan: # drop columns if there are any missing values
        # TODO handle missing values w/o dropping whole column
        text_df = text_df.dropna(axis=1, how='any')
        nan_dropped = get_dropped(headers, text_df.columns.values)
        nan_dropped = nan_dropped.difference(dtype_dropped)
        print('dropped columns with missing values: {0} \n'.format(list(nan_dropped)))
    
    data = {}
    print('normalizing headers')
    data['headers'] = normalize_headers(headers, model) 

    for col in text_df.columns.values:
        print('normalizing column: ', col)
        data[normalize_words(col, to_list=False)] = normalize_text(text_df[col].values, model) 

    return data


def get_timestamp():
    return datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')


def load_class_relationships(model, ontology_name='dbpedia_2016-10', prune=True):

    rels_file_name = get_relationships_file_name(ontology_name, prune)
    with open('ontologies/{0}'.format(rels_file_name), 'r') as f:  
        relationships = json.load(f)

    return normalize_classes(relationships, model)


def normalize_words(s, replace_chars = {'_': ' ', '-': ' '}, to_list=True):
    s = underscore(s)  # ow need lower
    for old, new in replace_chars.items():
        s = s.replace(old, new)
    if to_list:
        return s.split(' ')
    else:
        return s  

def in_vocab(word_list, model):
    if isinstance(word_list, str):
        word_list = word_list.split(' ')
    return all([word in model.wv.vocab for word in word_list])
        

def normalize_classes(relationships, model):
    # filter out keys with out-of-vocab words (strict by default -- all words in class name must be in vocab)
    relationships = {name: rels for (name, rels) in relationships.items() if in_vocab(name, model)}
    classes = relationships.keys()  # filtered class list

    # remove filtered classes from parent and child lists
    for name, rels in relationships: 
        rels['children'] = [cl in rels['children'] if (cl in classes)] 
        rels['parents'] = [cl in rels['parents'] if (cl in classes)] 

    return relationships

def normalize_headers(headers, model):
    headers = np.array([normalize_words(h) for h in headers])  # list of lists of single words

    in_vocab = [np.all([word in model.wv.vocab for word in h]) for h in headers]   
    print('dropped {0} out of {1} headers for having out-of-vocab words. \n'.format(len(headers) - sum(in_vocab), len(headers)))

    return headers[in_vocab]


def normalize_text(text, model):
    text = np.array([normalize_words(t) for t in text])  # list of lists of single words

    in_vocab = [np.all([word in model.wv.vocab for word in t]) for t in text]
    print('dropped {0} out of {1} text values for having out-of-vocab words \n'.format(len(text) - sum(in_vocab), len(text)))

    return text[in_vocab] 


def print_top_similarities(classes, similarities, n_keep=20):
    sort_inds = np.argsort(similarities)[::-1]
    similarities = similarities[sort_inds]
    classes = classes[sort_inds]

    print('top {0} classes, similarities: \n'.format(n_keep))
    for cl, sim in zip(classes[:n_keep], similarities[:n_keep]):
        print(cl, sim)
    print('\n\n')


class NumpyEncoder(json.JSONEncoder):
    
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)