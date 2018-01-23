import json

import numpy as np

from dataset_descriptor import DatasetDescriptor
from utils import mean_of_rows, path_to_name
from embedding import Embedding
from dataset import EmbeddedDataset
from class_tree import EmbeddedClassTree

def evaluate(scores, labels):
    scores = scores if isinstance(scores, np.ndarray) else np.array(scores)
    labels = labels if isinstance(labels, np.ndarray) else np.array(labels)

    assert labels.dtype == 'int'
    assert max(labels) == 1
    assert min(labels) == -1
    assert len(labels) == len(scores)

    results = {}
    pos_inds = np.where(labels == 1)[0]
    neg_inds = np.where(labels == -1)[0]
    
    results['positive'] = np.dot(scores[pos_inds], labels[pos_inds]) / len(pos_inds)
    results['negative'] = np.dot(scores[neg_inds], labels[neg_inds]) / len(neg_inds)
    
    return results


def get_labels(dataset_path, classes):
    dataset_path = dataset_path.split('.')[0]  # remove file extension
    labels_filename = '{0}_positive_examples.json'.format(dataset_path)
    with open(labels_filename) as f:
        positive_examples = json.load(f)
    
    return np.array([1 if cl in positive_examples else -1 for cl in classes])  

def keyval_str(key, val):
    return '{0}={1}'.format(key, val.__name__ if hasattr(val, '__name__') else str(val))


def config_string(model_config, dataset_path):
    config = {'dataset': path_to_name(dataset_path)}
    config.update(model_config)
    return ' '.join([ keyval_str(key, val) for (key, val) in config.items()])


# def run_trial(model_config=None, dataset=None, tree=None, embedding=None, verbose=True, max_num_samples=1e6):
def run_trial(trial_kwargs):
    labels = trial_kwargs.pop('labels')
    duke = DatasetDescriptor(**trial_kwargs)
    scores = duke.get_dataset_class_scores()
    return evaluate(scores, labels)


def main(
    tree_path='ontologies/class-tree_dbpedia_2016-10.json',
    embedding_path='embeddings/wiki2vec/en.model',
    dataset_paths=['data/185_baseball.csv'],
    model_configs=[{'row_agg_func': mean_of_rows, 'tree_agg_func': np.mean, 'source_agg_func': mean_of_rows}],
    max_num_samples=1e6,
    verbose=True,
    ):

    trial_results = {}

    embedding = Embedding(embedding_path=embedding_path, verbose=verbose)
    tree = EmbeddedClassTree(embedding, tree_path=tree_path, verbose=verbose)
    
    trial_kwargs = {
        'tree': tree,
        'embedding': embedding,
        'max_num_samples': max_num_samples,
        'verbose': verbose,
    }

    for dat_path in dataset_paths:
        trial_kwargs['dataset'] = EmbeddedDataset(embedding, dat_path, verbose=verbose)
        trial_kwargs['labels'] = get_labels(dat_path, tree.classes)

        for config in model_configs:
            trial_kwargs.update(config)
            trial_results[config_string(config, dat_path)] = run_trial(trial_kwargs)

    print('\n\nresults from evaluation trials:\n', trial_results, '\n\n')
    

if __name__ == '__main__':
    main()
