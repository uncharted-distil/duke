import re
import time

import numpy as np
import pandas as pd
from gensim.models import Word2Vec


def timeit(fn, args=None):
    start = time.time()
    print('calling function {0}'.format(fn.__name__))
    result = fn(*args) if args else fn()
    print('function {0} took {1} seconds'.format(fn.__name__, time.time() - start))
    return result


def load_model(model_name='wiki2vec'):
    ''' Load a word2vec model from a file in models/ '''
    models = {
        'wiki2vec': 'en_1000_no_stem/en.model',  # file paths hard coded here
    }
    start = time.time()
    model = Word2Vec.load("models/{0}".format(models[model_name]))
    return model


def flatten(lol):
    if np.any([isinstance(el, list) for el in lol]):
        return lol  # just return if already flat
    else:
        unpack = [el for sublist in lol for el in sublist]
        return flatten(unpack)  # if there are more layers to unpack, recurse

def normalize_types(types, model):
    # create a lol of types split by capitalization
    types = np.array([re.findall('[A-Z][^A-Z]*', typ) for typ in types])  # list of lists of single words
    # TODO more general processing? split by spaces?
    # remove types with out-of-vocab words

    in_vocab = [np.all([word in model.wv.vocab for word in typ]) for typ in types]   
    return types[in_vocab]


def normalize_headers(headers, model):
    headers = np.array([h.replace('_', ' ').lower().split(' ') for h in headers])  # list of lists of single words
    in_vocab = [np.all([word in model.wv.vocab for word in h]) for h in headers]   
    return headers[in_vocab]


def normalize_text(text, model):
    text = np.array([t.replace('_', ' ').lower().split(' ') for t in text]) # list of lists of single words
    in_vocab = [np.all([word in model.wv.vocab for word in t]) for t in text]
    return text[in_vocab] 


def get_similar_types(data, types, model):

    similarities = np.zeros(len(types))
    n_processed = 0
    for dat in data:
        try:
            if not np.all([d in model.wv.vocab for d in dat]):
                # skip with logging if not in vocab (should have been prevented by normalization)
                print('out of vocab: ', dat, [d in model.wv.vocab for d in dat])
            else:
                similarities += np.array([model.wv.n_similarity(dat, typ) for typ in types])
                n_processed += 1
        except KeyError as err:
            print('error checking distance of word {0} to types (out of vocab?):'.format(dat), err)
        except Exception as err:
            print('unknown error: ', err)
            raise err

    similarities /= max(1, n_processed)  # divide to get average 
    print('max, min similarities: ', np.max(similarities), ', ', np.min(similarities), '\n\n')

    # sort types by average similarity and unpack lists 
    sort_indices = np.argsort(similarities)[::-1]
    sorted_types = np.array(types)[sort_indices]
    sorted_similarities = similarities[sort_indices]

    return sorted_types, sorted_similarities


def load_dataset(dataset_name, model):
    csv_path = 'data/{0}/{0}_dataset/tables/learningData.csv'.format(dataset_name)
    full_df = pd.read_csv(csv_path, header=0)  # read csv assuming first line has header text
    text_df = full_df.select_dtypes(['object'])  # drop non-text rows (pandas strings are of type 'object')
    # TODO confirm that the columns selected can't be cast to a numeric type to avoid numeric strings (e.g. '1')
    
    # package data: concat then normalize headers and text columns
    all_headers = normalize_headers(full_df.columns.values, model)
    text = np.concatenate([text_df[h].values for h in text_df.columns])  # flatten content of all columns in word list
    text = normalize_text(text, model)
    
    return np.concatenate([all_headers, text])  


def load_types(model):
    # load types and normalize (remove out of vocab etc.)
    with open('models/types', 'r') as f:  
        types = f.read().splitlines()
        return normalize_types(types, model)  
        

def main(dataset_name = '185_baseball', n_keep = 20):
    
    # model = load_model()
    model = timeit(load_model)

    types = load_types(model)
    # print('examples from types: ', types[:10], ', ... , ', types[-10:], '\n\n')

    data = load_dataset(dataset_name, model)
    # print('examples from data: ', data[:10], ', ... , ', data[-10:], '\n\n')

    # sorted_types, similarities = get_similar_types(data, types, model)
    sorted_types, similarities = timeit(get_similar_types, [data, types, model])
    sorted_types = np.array([' '.join(typ) for typ in sorted_types])  # unpack lol with spaces between words and convert as np array
   
    print('top {0} types, similarities: \n'.format(n_keep))
    for typ, sim in zip(sorted_types[:n_keep], similarities[:n_keep]):
        print(typ, sim)



if __name__ == '__main__':
    main()
