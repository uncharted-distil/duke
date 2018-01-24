import json
import os
import pandas as pd
import numpy as np
import glob
from subprocess import call


def gen_label_stubs(ontology_path='../ontologies/class-tree_dbpedia_2016-10.json'):

    with open('{0}'.format(ontology_path), 'r') as f:  
        tree = json.load(f)
    
    classes = list(tree.keys())
    df = pd.DataFrame(classes, columns=['class'])
    df['label'] = ''

    for file_name in glob.glob('*.csv'):
        print('creating stub for', file_name)
        file_name = file_name.split('.')[0]  # remove file extension
        # file_name = file_name.split('_')[1]  # remove number_ prefix
        df.to_csv('{0}_labels.csv'.format(file_name), index=False)

def labels_to_positive_list():

    paths = glob.glob('*_labels.csv')

    for label_file in paths:
        label_df = pd.read_csv(label_file)

        dataset_name = label_file[:-11]
        pos_examples = label_df['class'][label_df['label'] == 1].tolist()

        # if labels have been provided, write pos examples to file
        if pos_examples:
            with open('{0}_positive_examples.json'.format(dataset_name), 'w') as json_file:
                json.dump(pos_examples, json_file, indent=4)




def fill_labels(dataset_name='185_baseball'):

    labels_filename = 'data/{0}_labels.csv'.format(dataset_name)
    labels = pd.read_csv(labels_filename)
    return labels.fillna(-1, downcast='infer')
    # labels.to_csv(labels_filename, index=False)


def lines_to_json(dataset_name='185_baseball'):
    with open('data/{0}_positive_examples'.format(dataset_name)) as f:
        lines = f.readlines()
        
    examples = [s.strip() for s in lines]
    with open('data/{0}_positive_examples.json'.format(dataset_name), 'w') as json_file:
        json.dump(examples, json_file, indent=4)


def flatten_data_directories():
    data_files = glob.glob('data/*/*_dataset/tables/learningData.csv')
    
    for file_path in data_files:
        dataset_name = os.path.dirname(file_path).split('/')[1]
        dataset_name = '_'.join(dataset_name.split('_')[2:])  # remove "LL0_number_" prefix 
        print('dataset name: ', dataset_name)
        call(['cp', file_path, dataset_name + '.csv'])

if __name__ == '__main__':
    # flatten_data_directories()
    # gen_label_stubs()
    labels_to_positive_list()
