import sys
import json
import time
from datetime import datetime
from random import shuffle
import pandas as pd
import numpy as np

from dataset_description import DatasetDescriptor
from SentenceProducer import SentenceProducer

from similarity_functions import (freq_nearest_similarity,
                                  get_class_similarities, w2v_similarity)

from trees import tree_score

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

    return duke.get_description(dataset=dataset)

    
    # # EXAMPLE CONFIGURATION
    # sentenceProducer = SentenceProducer(model_name='/data/duke/models/word2vec/en_1000_no_stem/en.model', 
    #                     types_filename='/data/duke/models/ontologies/types',
    #                     type_hierarchy_filename='/data/duke/ontologies/type_hierarchy.json',
    #                     inverted_type_hierarchy_filename='/data/duke/ontologies/inverted_type_hierarchy.json',
    #                     verbose=False)
    # return sentenceProducer.produceSentenceFromDataframe(full_df)

if __name__ == '__main__':
    main(dataset=sys.argv[1], embedding_path=sys.argv[2])