import cProfile as profile
import sys

import numpy as np

from dataset_descriptor import DatasetDescriptor


def main(
    dataset='185_baseball',
    embedding_path='en_1000_no_stem/en.model',  # wiki2vec model
    ontology_path='dbpedia_2016-10',
    row_agg_func=mean_of_rows,
    tree_agg_func=np.mean,
    source_agg_func=mean_of_rows,
    max_num_samples = 1e6,
    verbose=True,
    ):

    duke = DatasetDescriptor(
        dataset=dataset,
        embedding_path=embedding_path,
        ontology_path=ontology_path,
        similarity_func=similarity_func,
        tree_agg_func=tree_agg_func,
        source_agg_func=source_agg_func,
        max_num_samples=max_num_samples,
        verbose=verbose,
        )

    print('initialized duke dataset descriptor \n')

    return duke.get_description(dataset=dataset)


if __name__ == '__main__':
    main()
    # profile.run('main()', sort='time')
    # main(dataset=sys.argv[1], embedding_path=sys.argv[2])
