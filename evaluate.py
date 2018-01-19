import json

import numpy as np

from dataset_description import DatasetDescriptor
from similarity_functions import w2v_similarity


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

    print('\n\nevaluation:', evaluate(scores, labels))


if __name__ == '__main__':
    main()
