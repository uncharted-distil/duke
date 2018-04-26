import cProfile as profile
import sys

import numpy as np
from Duke.agg_functions import *
from Duke.dataset_descriptor import DatasetDescriptor
from Duke.utils import mean_of_rows


def main(
    dataset_path='data/185_baseball.csv',
    tree_path='ontologies/class-tree_dbpedia_2016-10.json',
    embedding_path='embeddings/wiki2vec/en.model',
    row_agg_func=mean_of_rows,
    tree_agg_func=parent_children_funcs(np.mean, max),
    source_agg_func=mean_of_rows,
    max_num_samples = 1e6,
    verbose=True,
    ):

    duke = DatasetDescriptor(
        dataset=dataset_path,
        tree=tree_path,
        embedding=embedding_path,
        row_agg_func=row_agg_func,
        tree_agg_func=tree_agg_func,
        source_agg_func=source_agg_func,
        max_num_samples=max_num_samples,
        verbose=verbose,
        )

    print('initialized duke dataset descriptor \n')

    return duke.get_dataset_description()


if __name__ == '__main__':
    main()
    # profile.run('main()', sort='time')
    # main(dataset=sys.argv[1], embedding_path=sys.argv[2])
