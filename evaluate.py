import json
import os

import numpy as np

from dataset_descriptor import DatasetDescriptor
from utils import mean_of_rows, path_to_name

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


def main(
    dataset_path='data/185_baseball.csv',
    ontology_path='ontologies/dbpedia_2016-10.nt',
    embedding_path='embeddings/wiki2vec/en.model',
    row_agg_func=mean_of_rows,
    tree_agg_func=np.mean,
    source_agg_func=mean_of_rows,
    max_num_samples = 1e6,
    verbose=True,
    ):

    duke = DatasetDescriptor(
        dataset=dataset_path,
        ontology=ontology_path,
        embedding=embedding_path,
        row_agg_func=row_agg_func,
        tree_agg_func=tree_agg_func,
        source_agg_func=source_agg_func,
        max_num_samples=max_num_samples,
        verbose=verbose,
        )

    print('initialized duke dataset descriptor \n')

    scores = duke.get_dataset_class_scores()
    labels_filename = 'data/{0}_positive_examples.json'.format(path_to_name(dataset_path))
    with open(labels_filename) as f:
        positive_examples = json.load(f)
    
    classes = np.array(duke.classes)
    labels = np.array([1 if cl in positive_examples else -1 for cl in classes])    

    print('\n\nevaluation:', evaluate(scores, labels))


if __name__ == '__main__':
    main()
