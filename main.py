import sys
import json
import time
from datetime import datetime
from random import shuffle
import pandas as pd
import numpy as np

from dataset_description import DatasetDescriptor

from similarity_functions import (freq_nearest_similarity,
                                  get_class_similarities, w2v_similarity)

from trees import tree_score


def main(
    dataset='185_baseball',
    embedding_path='en_1000_no_stem/en.model',  # wiki2vec model
    ontology_path='dbpedia_2016-10',
    similarity_func=w2v_similarity,
    tree_agg_func=np.mean,
    source_agg_func=lambda scores: np.mean(scores, axis=0),
    max_num_samples = 2000,
    verbose=True,
    ):

    duke = DatasetDescriptor(
        # dataset=dataset,
        dataset=None,
        embedding_path=embedding_path,
        ontology_path=ontology_path,
        similarity_func=similarity_func,
        tree_agg_func=tree_agg_func,
        source_agg_func=source_agg_func,
        max_num_samples=max_num_samples,
        verbose=verbose,
        )

    print('loaded duke object')

    return duke.get_description(dataset=dataset)
    # print('duke produces description:', description)

    

if __name__ == '__main__':
    main()
# def run_trial(data, classes, model, similarity_func=w2v_similarity, extra_args=None, max_num_samples=None):

#     print('running trial with similarity function: {0}{1}\n'.format(similarity_func.__name__, ', and extra args: {0}'.format(extra_args) if extra_args else ''))
    
#     similarities = {}
#     for source, words in data.items():
        
#         if max_num_samples and len(words) > max_num_samples:
#             print('subsampling word list of length {0} to {1}'.format(len(words), max_num_samples))
#             shuffle(words)
#             words = words[:max_num_samples]  
        
#         print('computing class similarity using source: ', source)
#         similarities[source] = get_class_similarities(words, classes, model, similarity_func, extra_args)

#     return similarities



# def similarity_trials(
#     model_name='wiki2vec',
#     # dataset_name='LL0_1037_ada_prior',
#     dataset_name='185_baseball',
#     max_num_samples = 2000,
#     configs = [
#         {'similarity_function': w2v_similarity},
#         # {'similarity_function': freq_nearest_similarity},
#         # {'similarity_function': freq_nearest_similarity, 'extra_args': {'n_nearest': 3}},
#     ]):
    
#     print('\n loading model')
#     model = load_model(model_name)
#     # TODO print model stats
    
#     print('\n loading classes and relationship heirarchy')
#     relationships = load_class_relationships(model)
#     classes = list(relationships.keys())
    
#     print('\n loading dataset \n')
#     data = load_dataset(dataset_name, model)
#     # TODO print dataset stats

#     for conf in configs:
#         sim_func = conf['similarity_function']
#         extra_args = conf.get('extra_args')

#         similarities = run_trial(data, classes, model, sim_func, extra_args, max_num_samples)
        
#         # write results of trial to file along with trial config info
#         record = {
#             'model': model_name, 
#             'classes': classes, 
#             'dataset': dataset_name,
#             'sim_func': sim_func.__name__, 
#             'extra_args': extra_args,
#             'similarities': similarities,  # dict mapping column headers to similarity vectors (in the same order as classes list)
#             }
        
#         print('\n writing trial results to file')
#         with open('trials/trial_{0}.json'.format(get_timestamp()), 'w') as f:
#             json.dump(record, f, cls=NumpyEncoder, indent=4)
#             print('trial results saved \n\n')

#         for source, scores in similarities.items():
#             print('***** similarity trial results for source {0}:\n'.format(source))
#             score_map = dict(zip(classes, scores))
#             print_tree_results(score_map, relationships)


# def print_tree_results(score_map, tree):

#     classes = list(tree.keys())
#     scores = [score_map[cl] for cl in classes]
    
#     print('using base score: \n')
#     print_top_similarities(classes, scores)

#     max_tree_score = aggregate_score(score_map, tree, max)
#     print('using max tree score: \n')
#     print_top_similarities(classes, max_tree_score)
    
#     avg_tree_score = aggregate_score(score_map, tree, np.mean)
#     print('using avg tree score: \n')
#     print_top_similarities(classes, avg_tree_score)


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
    

# def main(model_name='wiki2vec', dataset_name='LL0_1037_ada_prior'):

#     # EXAMPLE CONFIGURATION
#     ps = ProduceSentence(model_name='/data/duke/models/word2vec/en_1000_no_stem/en.model', 
#                         types_filename='/data/duke/models/ontologies/types',
#                         type_heirarchy_filename='type_heirarchy.json')

#     csv_path = '/data/duke/data/{0}/{0}_dataset/tables/learningData.csv'.format(dataset_name)
#     full_df = pd.read_csv(csv_path, header=0)  # read csv assuming first line has header text. TODO handle files w/o headers

#     return ps.produceSentenceFromDataframe(full_df)



    # print('\n loading embedding model')
    # embedding = load_embedding(embedding_name)
    # # TODO print embedding stats
    
    # print('\n loading classes and relationship heirarchy')
    # relationships = load_class_relationships(embedding)
    # classes = list(relationships.keys())
    
    # print('\n loading dataset \n')
    # data = load_dataset(dataset_name, embedding)
    # # TODO print dataset stats

    # for conf in configs:
    #     sim_func = conf['similarity_function']
    #     extra_args = conf.get('extra_args')

    #     similarities = run_trial(data, classes, embedding, sim_func, extra_args, max_num_samples)
        
    #     # write results of trial to file along with trial config info
    #     record = {
    #         'embedding': embedding_name, 
    #         'classes': classes, 
    #         'dataset': dataset_name,
    #         'sim_func': sim_func.__name__, 
    #         'extra_args': extra_args,
    #         'similarities': similarities,  # dict mapping column headers to similarity vectors (in the same order as classes list)
    #         }
        
    #     print('\n writing trial results to file')
    #     with open('trials/trial_{0}.json'.format(get_timestamp()), 'w') as f:
    #         json.dump(record, f, cls=NumpyEncoder, indent=4)
    #         print('trial results saved \n\n')

    #     for source, scores in similarities.items():
    #         print('***** similarity trial results for source {0}:\n'.format(source))
    #         score_map = dict(zip(classes, scores))
    #         print_tree_results(score_map, relationships)