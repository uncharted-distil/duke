import glob
import itertools
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb

from class_tree import EmbeddedClassTree
from dataset import EmbeddedDataset
from dataset_descriptor import DatasetDescriptor
from embedding import Embedding
from utils import get_timestamp, max_of_rows, mean_of_rows, path_to_name
from agg_functions import null_tree_agg, maxabs_of_rows, maxabs, meanmax_tree_agg
from plotting import plot_heatmap, plot_horizontal_barplot


def check_args(scores, labels):
    scores = scores if isinstance(scores, np.ndarray) else np.array(scores)
    labels = labels if isinstance(labels, np.ndarray) else np.array(labels)

    assert labels.dtype == 'int'
    assert max(labels) == 1
    assert min(labels) == -1
    assert len(labels) == len(scores)

    return scores, labels


def evaluate(scores, labels, n_keep=None):
    scores, labels = check_args(scores, labels)

    results = {}
    neg_inds = np.where(labels == -1)[0]
    pos_inds = np.where(labels == 1)[0]
    
    # compute pos match rate
    n_pos = len(pos_inds)
    n_keep = n_keep if n_keep else len(pos_inds)
    top_inds = np.argsort(scores)[::-1][:n_keep]
    n_matching = len(set.intersection(set(pos_inds), set(top_inds)))
    results['positive_match_rate'] = n_matching / min(n_pos, n_keep)
    
    # average scores for negative and positive examples
    results['avg_positive_score'] = np.dot(scores[pos_inds], labels[pos_inds]) / len(pos_inds)
    results['avg_negative_score'] = np.dot(scores[neg_inds], labels[neg_inds]) / len(neg_inds)
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


def run_trial(trial_kwargs, labels, n_keep=3):
    duke = DatasetDescriptor(**trial_kwargs)
    scores = duke.get_dataset_class_scores()
    return evaluate(scores, labels, n_keep=n_keep)


def run_experiment(
    tree_path='ontologies/class-tree_dbpedia_2016-10.json',
    embedding_path='embeddings/wiki2vec/en.model',
    dataset_paths=['data/185_baseball.csv'],
    model_configs=[{'row_agg_func': mean_of_rows, 'tree_agg_func': np.mean, 'source_agg_func': mean_of_rows}],
    max_num_samples=int(1e5),
    n_keep=30,
    verbose=True,
    ):

    print('\nrunning evaluation trials using {0} datasets:\n{1}'.format(len(dataset_paths), '\n'.join(dataset_paths)))
    print('\nand {0} configs:\n{1}\n\n'.format(
        len(model_configs),
        '\n'.join(
            [
                ', '.join(['{0}: {1}'.format(key, func_name_str(val)) for (key, val) in config.items()]) 
                for config in model_configs
            ]
        )
    ))

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
        labels = get_labels(dat_path, tree.classes)

        for config in model_configs:
            print('\nrunning trial with config:', {key: func_name_str(val) for (key, val) in config.items()})
            # run trial using config
            trial_kwargs.update(config)
            trial_results = run_trial(trial_kwargs, labels, n_keep=n_keep)

            # add config and dataset name to results and append results to rows list
            trial_results.update({key: func_name_str(val) for (key, val) in config.items()})
            trial_results.update({'dataset': path_to_name(dat_path)})
            rows.append(trial_results)

    # print('\n\nresults from evaluation trials:\n', rows, '\n\n')

    df = pd.DataFrame(rows)
    df.to_csv('trials/trial_{0}.csv'.format(get_timestamp()), index=False)

    return df
    

def all_labeled_test(n_keep=7):

    agg_func_combs = itertools.product(
        [mean_of_rows, max_of_rows],  # , maxabs_of_rows],  # row agg funcs
        [np.mean, max, meanmax_tree_agg],    # tree agg funcs
        [mean_of_rows, max_of_rows],   # source agg funcs
    )

    # create dict list of all func combinations
    model_configs = [{'row_agg_func': row, 'tree_agg_func': tree, 'source_agg_func': source} for (row, tree, source) in agg_func_combs]

    # get all datasets with labels in the data/ directory
    dataset_paths = glob.glob('data/*_positive_examples.json')
    dataset_paths = [path.replace('_positive_examples.json', '.csv') for path in dataset_paths]

    df = run_experiment(
        dataset_paths=dataset_paths,
        model_configs=model_configs,
        n_keep=n_keep,
        )

    plot_results(df, n_keep)
    

def config_to_legend_string(config):
    return ' '.join(['{0}={1}'.format(key.split('_')[0], func_name_str(val).replace('_', ' ')) for (key, val) in config.items()])


def post_process(df):
    print('\npost-processing data')
    
    #compute score gap and negative of neg avg score
    df['score_gap'] = df['avg_positive_score'] - df['avg_negative_score']
    df['-avg_negative_score'] = - df['avg_negative_score']

    # populate config string column (used to identify configs in plots)
    df['config'] = ['{0}, {1}, {2}'.format(
            func_name_str(row['row_agg_func']).split('_')[0],
            func_name_str(row['tree_agg_func']).split('_')[0],
            func_name_str(row['source_agg_func']).split('_')[0],
            ) for index, row in df.iterrows()]


def plot_results(trial_results=None, n_keep=None):

    if trial_results is None:
        files = glob.glob('trials/*.csv')
        most_recent = sorted(files)[-1]  # assumes timestamp file suffix
        df = pd.read_csv(most_recent)

    elif isinstance(trial_results, str):
        df = pd.read_csv(trial_results)
        
    else:
        assert isinstance(trial_results, pd.DataFrame)
        df = trial_results

    post_process(df)

    sb.set_color_codes("muted")

    image_path = 'plots/barplot_score_gap_{0}.png'.format(get_timestamp())
    plot_horizontal_barplot(df, image_path=image_path, group_by='config', values=['score_gap', 'avg_positive_score'], colors=['b','g'])

    image_path = 'plots/barplot_positive_match_rate{0}_{1}.png'.format('_keep' + n_keep if n_keep else '', get_timestamp())
    plot_horizontal_barplot(df, image_path=image_path, group_by='config', values='positive_match_rate')

    image_path = 'plots/heatmap_positive_match_rate{0}_{1}.png'.format('_keep' + n_keep if n_keep else '', get_timestamp())
    plot_heatmap(df, image_path=image_path, columns='config', index='dataset', values='positive_match_rate')
    

if __name__ == '__main__':
    # all_labeled_test()
    plot_results()
