import cProfile as profile
import numpy as np

from dataset_description import DatasetDescriptor
from similarity_functions import w2v_similarity


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

    print('initialized duke dataset descriptor \n')

    print(duke.get_description(dataset))


if __name__ == '__main__':
    main()
    # profile.run('main()', sort='time')
