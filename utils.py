# import re
# import time
# from datetime import datetime
# import json
# from inflection import underscore

# import numpy as np
# import pandas as pd
# from gensim.models import Word2Vec

# from ontologies.ontology import get_relationships_file_name


# def timeit(func, args=None):
#     start = time.time()
#     log('calling {0} \n'.format(func.__name__))
#     result = func(*args) if args else func()
#     log('{0} took {1} seconds \n\n'.format(func.__name__, time.time() - start))
#     return result


# def load_model(model_name='/data/duke/models/word2vec/en_1000_no_stem/en.model'):
#     ''' Load a word2vec model from a file in models/ '''
#     return Word2Vec.load(model_name)


# def get_dropped(all_headers, new_headers):
#     return set(all_headers).difference(set(new_headers))


# def get_timestamp():
#     return datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')


# def load_ontology(model, ontology_path='dbpedia_2016-10', prune=True):

#     rels_file_name = get_relationships_file_name(ontology_path, prune)
#     with open('ontologies/{0}'.format(rels_file_name), 'r') as f:  
#         relationships = json.load(f)

#     return normalize_classes(relationships, model)


# def normalize_words(s, replace_chars = {'_': ' ', '-': ' '}, to_list=True):
#     s = underscore(s)  # ow need lower
#     for old, new in replace_chars.items():
#         s = s.replace(old, new)
#     if to_list:
#         return s.split(' ')
#     else:
#         return s  

# def in_vocab(word_list, model):
#     if isinstance(word_list, str):
#         word_list = word_list.split(' ')
#     return all([word in model.wv.vocab for word in word_list])
        

# def normalize_classes(relationships, model):
#     # filter out keys with out-of-vocab words -- all words in class name must be in vocab
#     relationships = {name: rels for (name, rels) in relationships.items() if in_vocab(name, model)}
#     classes = list(relationships.keys())  # filtered class list
#     #     log('dropped {0} out of {1} type values for having out-of-vocab words. \n'.format()

#     # remove filtered classes from parent and child lists
#     for name, rels in relationships.items(): 
#         rels['children'] = [cl for cl in rels['children'] if (cl in classes)] 
#         rels['parents'] = [cl for cl in rels['parents'] if (cl in classes)] 

#     return relationships


# # def normalize_headers(headers, model):
# #     headers = np.array([normalize_words(h) for h in headers])  # list of lists of single words

# # def normalize_headers(headers, model, verbose=False):
# #     headers = np.array([h.replace('_', ' ').replace('-', ' ').lower().split(' ') for h in headers])  # list of lists of single words

# #     in_vocab = [np.all([word in model.wv.vocab for word in h]) for h in headers]   
# #     if(verbose):
# #         print('dropped {0} out of {1} headers for having out-of-vocab words. \n'.format(len(headers) - sum(in_vocab), len(headers)))

# #     return headers[in_vocab]


# def normalize_text(text, model):
#     text = np.array([normalize_words(t) for t in text])  # list of lists of single words

# # def normalize_text(text, model, verbose=False):
# #     text = np.array([t.replace('_', ' ').replace('-', ' ').lower().split(' ') for t in text])  # list of lists of single words

# #     in_vocab = [np.all([word in model.wv.vocab for word in t]) for t in text]
# #     if(verbose):
# #         print('dropped {0} out of {1} text values for having out-of-vocab words \n'.format(len(text) - sum(in_vocab), len(text)))

# #     return text[in_vocab] 


# def print_top_similarities(classes, score, n_keep=20):


#     # convert lists, dicts to np arrays
#     if isinstance(classes, list): 
#         classes = np.array(classes)
#     if isinstance(score, list): 
#         score = np.array(score)
#     if isinstance(score, dict):
#         score = np.array([score[cl] for cl in classes])

#     sort_inds = np.argsort(score)[::-1]
#     score = score[sort_inds]
#     classes = classes[sort_inds]

#     log('top {0} classes, score: \n'.format(n_keep))
#     for cl, sim in zip(classes[:n_keep], score[:n_keep]):
#         log(cl,': ', sim)
#     log('\n\n')


# class NumpyEncoder(json.JSONEncoder):
    
#     def default(self, obj):
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()
#         return json.JSONEncoder.default(self, obj)