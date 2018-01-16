import sys
import json
import time
import pandas as pd

from ProduceSentence import ProduceSentence


def main(model_name='wiki2vec', dataset_name='LL0_1037_ada_prior'):

    # EXAMPLE CONFIGURATION
    ps = ProduceSentence(model_name='/data/duke/models/word2vec/en_1000_no_stem/en.model', 
                        types_filename='/data/duke/models/ontologies/types',
                        type_heirarchy_filename='type_heirarchy.json',
                        verbose=True)

    csv_path = '/data/duke/data/{0}/{0}_dataset/tables/learningData.csv'.format(dataset_name)
    full_df = pd.read_csv(csv_path, header=0)  # read csv assuming first line has header text. TODO handle files w/o headers

    return ps.produceSentenceFromDataframe(full_df)


if __name__ == '__main__':
    print(main(dataset_name=sys.argv[1]))
