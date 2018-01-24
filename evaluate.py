import json
import glob

import numpy as np
import pandas as pd

from dataset_descriptor import DatasetDescriptor
from utils import mean_of_rows, path_to_name, get_timestamp
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
    
    # average scores for negative and positive examples
    results['avg_positive_score'] = np.dot(scores[pos_inds], labels[pos_inds]) / len(pos_inds)
    results['avg_negative_score'] = -np.dot(scores[neg_inds], labels[neg_inds]) / len(neg_inds)
    results['n_positive_samples'] = len(pos_inds)
    results['n_negative_samples'] = len(neg_inds)
    
    return results


def get_labels(dataset_path, classes):
    dataset_path = dataset_path.split('.')[0]  # remove file extension
    labels_filename = '{0}_positive_examples.json'.format(dataset_path)
    with open(labels_filename) as f:
        positive_examples = json.load(f)
    
    return np.array([1 if cl in positive_examples else -1 for cl in classes])  


def func_name_str(func):
    return func.__name__ if hasattr(func, '__name__') else str(func)


# def run_trial(model_config=None, dataset=None, tree=None, embedding=None, verbose=True, max_num_samples=1e6):
def run_trial(trial_kwargs, labels):
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

    print('\nrunning evaluation trials using datasets:\n', dataset_paths)
    print('and configs:', [{key: func_name_str(val) for (key, val) in config.items()} for config in model_configs])


    embedding = Embedding(embedding_path=embedding_path, verbose=verbose)
    tree = EmbeddedClassTree(embedding, tree_path=tree_path, verbose=verbose)
    
    trial_kwargs = {
        'tree': tree,
        'embedding': embedding,
        'max_num_samples': max_num_samples,
        'verbose': verbose,
    }

    rows = []
    for dat_path in dataset_paths:
        print('\nloading dataset:', dat_path)
        trial_kwargs['dataset'] = EmbeddedDataset(embedding, dat_path, verbose=verbose)
        # trial_kwargs['labels'] = get_labels(dat_path, tree.classes)
        labels = get_labels(dat_path, tree.classes)

        for config in model_configs:
            print('\nrunning trial with config:', {key: func_name_str(val) for (key, val) in config.items()})
            # run trial using config
            trial_kwargs.update(config)
            print('trial kwargs:', trial_kwargs)
            trial_results = run_trial(trial_kwargs, labels)            

            # add config and dataset name to results and append results to rows list
            trial_results.update({key: func_name_str(val) for (key, val) in config.items()})
            trial_results.update({'dataset': path_to_name(dat_path)})
            rows.append(trial_results)

    print('\n\nresults from evaluation trials:\n', rows, '\n\n')

    df = pd.DataFrame(rows)
    df.to_csv('trials/trial_{0}.csv'.format(get_timestamp()), index=False)
    

def all_labeled_test():
    model_configs = [
        {'row_agg_func': mean_of_rows, 'tree_agg_func': np.mean, 'source_agg_func': mean_of_rows},
        {'row_agg_func': mean_of_rows, 'tree_agg_func': max, 'source_agg_func': mean_of_rows},
    ]

    dataset_paths = glob.glob('data/*_positive_examples.json')
    dataset_paths = [path.replace('_positive_examples.json', '.csv') for path in dataset_paths]

    main(
        dataset_paths=dataset_paths,
        model_configs=model_configs,
        )


if __name__ == '__main__':
    all_labeled_test()