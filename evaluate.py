import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from inflection import pluralize, underscore

from dataset_description import DatasetDescriptor
from ontologies.ontology import get_tree_file_name
from similarity_functions import w2v_similarity
from trees import tree_score

# load dataset labels
# evaluate performance for score

def evaluate(scores, labels):
    scores = scores if isinstance(scores, np.ndarray) else np.array(scores)
    labels = labels if isinstance(labels, np.ndarray) else np.array(labels)

    assert(labels.dtype == 'int')
    assert(max(labels) == 1)
    assert(min(labels) == -1)
    assert(len(labels) == len(scores))

    results = {}
    pos_inds = np.where(labels == 1)[0]
    neg_inds = np.where(labels == -1)[0]
    
    results['positive'] = np.dot(scores[pos_inds], labels[pos_inds]) / len(pos_inds)
    results['negative'] = np.dot(scores[neg_inds], labels[neg_inds]) / len(neg_inds)
    # results['combined'] = np.dot(scores, labels) / len(labels)
    
    return results

# def evaluate_descriptor(descriptor, dataset_name='185_baseball'):

#     with open('ontologies/{0}'.format(tree_file_name), 'r') as f:  
#         tree = json.load(f)



def gen_label_stub(ontology_path='dbpedia_2016-10', prune=False):

    tree_file_name = get_tree_file_name(ontology_path, prune)
    with open('ontologies/{0}'.format(tree_file_name), 'r') as f:  
        tree = json.load(f)

    classes = list(tree.keys())
    df = pd.DataFrame(classes, columns=['class'])
    df.to_csv('data/class_label_stub.csv', index=False)


# def labels_to_related(dataset_name='185_baseball'):
#     labels_filename = 'data/{0}_labels.csv'.format(dataset_name)
#     labels = pd.read_csv(labels_filename)
#     related = list(labels['class'].values)

#     with open('data/{0}_related.csv'.format(dataset_name), 'w') as f:
#         json.dump(related, f)

# def load_related(dataset_name='185_baseball'):
#     with open('data/{0}_related.csv'.format(dataset_name), 'r') as f:
#         return json.load(f)

def fill_labels(dataset_name='185_baseball'):

    labels_filename = 'data/{0}_labels.csv'.format(dataset_name)
    labels = pd.read_csv(labels_filename)
    return labels.fillna(-1, downcast='infer')
    # labels.to_csv(labels_filename, index=False)

def lines_to_json(dataset_name='185_baseball'):
    with open('data/{0}_positive_examples'.format(dataset_name)) as f:
        lines = f.readlines()
    examples = [s.strip() for s in lines]
    with open('data/{0}_positive_examples.json'.format(dataset_name), 'w') as f:
        json.dump(examples, f)

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
        dataset=None,
        embedding_path=embedding_path,
        ontology_path=ontology_path,
        similarity_func=similarity_func,
        tree_agg_func=tree_agg_func,
        source_agg_func=source_agg_func,
        max_num_samples=max_num_samples,
        verbose=verbose,
        )

    print('initialized duke dataset descriptor \n')

    scores = duke.get_dataset_class_scores(dataset, reset_scores=True)

    labels_filename = 'data/{0}_positive_examples.json'.format(dataset)
    with open(labels_filename) as f:
        positive_examples = json.load(f)

    
    classes = np.array(duke.classes)
    labels = np.array([1 if cl in positive_examples else -1 for cl in classes])    
    # # labels_filename = 'data/{0}_labels.csv'.format(dataset)
    # # labels_df = pd.read_csv(labels_filename)
    # # labels_df.fillna(-1, downcast='infer', inplace=True)  

    # label_map = dict(zip(labels_df['class'], labels_df['label']))
    # labels = [label_map[cl] for cl in duke.classes]  # remove labels not in duke's classes (out of vocab)
    # print('num duke classes:', len(classes))
    # print('num labels:', len(labels))
    # # assert(all(classes == np.array(duke.classes)))  
    # pos_inds = np.where(labels == 1)[0]
    # print('classes related to {0}: {1} \n'.format(dataset, classes[pos_inds]))

    print('\n\nevaluation:', evaluate(scores, labels))


# plot evaluation for different methods, variants


if __name__ == '__main__':
    main()
    # gen_label_stub()
    # fill_labels()
    # lines_to_json()
