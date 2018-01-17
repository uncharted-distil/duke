import sys
import json
import time
import pandas as pd

from SentenceProducer import SentenceProducer


def main(model_name='wiki2vec', dataset_name='LL0_1037_ada_prior'):

    # EXAMPLE CONFIGURATION
    sentenceProducer = SentenceProducer(model_name='/data/duke/models/word2vec/en_1000_no_stem/en.model', 
                        types_filename='/data/duke/models/ontologies/types',
                        type_hierarchy_filename='/data/duke/ontologies/type_hierarchy.json',
                        inverted_type_hierarchy_filename='/data/duke/ontologies/inverted_type_hierarchy.json',
                        verbose=False)

    csv_path = '/data/duke/data/{0}/{0}_dataset/tables/learningData.csv'.format(dataset_name)
    full_df = pd.read_csv(csv_path, header=0)  # read csv assuming first line has header text. TODO handle files w/o headers

    return sentenceProducer.produceSentenceFromDataframe(full_df)


if __name__ == '__main__':
    print(main(dataset_name=sys.argv[1]))
