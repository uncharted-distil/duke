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


def run_trial(trial_kwargs, labels):
    duke = DatasetDescriptor(**trial_kwargs)
    scores = duke.get_dataset_class_scores()
    return evaluate(scores, labels)


def run_experiment(
    tree_path='ontologies/class-tree_dbpedia_2016-10.json',
    embedding_path='embeddings/wiki2vec/en.model',
    dataset_paths=['data/185_baseball.csv'],
    model_configs=[{'row_agg_func': mean_of_rows, 'tree_agg_func': np.mean, 'source_agg_func': mean_of_rows}],
    max_num_samples=1e6,
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
            trial_results = run_trial(trial_kwargs, labels)            

            # add config and dataset name to results and append results to rows list
            trial_results.update({key: func_name_str(val) for (key, val) in config.items()})
            trial_results.update({'dataset': path_to_name(dat_path)})
            rows.append(trial_results)

    # print('\n\nresults from evaluation trials:\n', rows, '\n\n')

    df = pd.DataFrame(rows)
    df.to_csv('trials/trial_{0}.csv'.format(get_timestamp()), index=False)

    return df
    

def all_labeled_test():

    agg_func_combs = itertools.product(
        [mean_of_rows, max_of_rows],  # row agg funcs
        [np.mean, max],    # tree agg funcs
        [mean_of_rows, max_of_rows],   # source agg funcs
    )

    # create dict list of all func combinations
    model_configs = [{'row_agg_func': row, 'tree_agg_func': tree, 'source_agg_func': source} for (row, tree, source) in agg_func_combs]
    ## manually set config list
    # model_configs = [
    #     {'row_agg_func': mean_of_rows, 'tree_agg_func': np.mean, 'source_agg_func': mean_of_rows},
    # ]

    dataset_paths = glob.glob('data/*_positive_examples.json')
    dataset_paths = [path.replace('_positive_examples.json', '.csv') for path in dataset_paths]

    df = run_experiment(
        dataset_paths=dataset_paths,
        model_configs=model_configs,
        )

    plot_results(df)
    

def config_to_legend_string(config):
    return ' '.join(['{0}={1}'.format(key.split('_')[0], func_name_str(val).replace('_', ' ')) for (key, val) in config.items()])


def get_config_string_col(df):
    # return ['row={0} tree={1} source={2}'.format(
    return ['{0}, {1}, {2}'.format(
            func_name_str(row['row_agg_func']).split('_')[0],
            func_name_str(row['tree_agg_func']).split('_')[0],
            func_name_str(row['source_agg_func']).split('_')[0],
            ) for index, row in df.iterrows()]


def plot_results(trial_results=None, n_top=5):

    if trial_results is None:
        files = glob.glob('trials/*.csv')
        most_recent = sorted(files)[-1]  # assumes timestamp file suffix
        df = pd.read_csv(most_recent)

    elif isinstance(trial_results, str):
        df = pd.read_csv(trial_results)
        
    else:
        assert isinstance(trial_results, pd.DataFrame)
        df = trial_results

    print('\npost-processing data')
    df['config'] = get_config_string_col(df)
    df['score_gap'] = df['avg_positive_score'] - df['avg_negative_score']
    df['-avg_negative_score'] = - df['avg_negative_score']

    config_score_map = df.groupby('config')['score_gap'].mean()     
    config_strings = np.array(list(config_score_map.keys()))
    config_scores = config_score_map.values
    sort_inds = np.argsort(config_scores)[::-1]
    top_scores = config_scores[sort_inds][:n_top]
    top_configs = config_strings[sort_inds][:n_top]
    print('\n\ntop {0} configs, scores:\n{1}\n\n'.format(
        n_top,
        '\n'.join([str(x) for x in list(zip(top_configs, top_scores))])
        ))

    print('plotting config scores\n')
    fig = plt.figure(figsize=(14, 6))
    grid = plt.GridSpec(1, 3)  # , hspace=0.2, wspace=0.2)
    ax0 = fig.add_subplot(grid[0, 0])
    ax1 = fig.add_subplot(grid[0, 1], sharey=ax0)
    ax2 = fig.add_subplot(grid[0, 2], sharey=ax0)

    sb.barplot(x='config', y='score_gap', data=df, ax=ax0)
    sb.barplot(x='config', y='avg_positive_score', data=df, ax=ax1)
    sb.barplot(x='config', y='-avg_negative_score', data=df, ax=ax2)

    plt.savefig('plots/scores_{0}.png'.format(get_timestamp()))


if __name__ == '__main__':
    all_labeled_test()
    # plot_results()
