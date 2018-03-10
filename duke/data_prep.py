import pandas as pd
from glob import glob
import json
import scipy.sparse as sp
import os.path
import numpy as np
import numpy.random as rn


def prep_dataset(dataset):
    return dataset


def read_json(json_path):
    with open(json_path) as json_file:
        return json.load(json_file)


def write_json(obj, json_path):
    with open(json_path, 'w') as json_file:
        return json.dump(obj, json_file, indent=2)


def vectorize_labels(labels, label_to_index):
    n_labels = len(labels)
    vals = np.ones(n_labels, dtype=bool)
    inds = np.array([label_to_index[label] for label in labels])
    return sp.csr_matrix((vals, (np.zeros(n_labels), inds)))


def vectorize_dataset(dataset, feat_map):
    # return list of sparse n_columns x n_features matrices
    return [feat_map(col) for col in dataset.T]


def rand_feat_map(col, n_features=100, n_nz=10):
    # ignores column data, returns random sparse binary row vector
    indices = rn.randint(n_features, size=n_nz)
    indptr = [0, n_nz]
    data = np.ones(n_nz, dtype=bool)
    return sp.csr_matrix((data, indices, indptr), shape=(1, n_features))


def normalize_label(label):
    # TODO part of vocab/label config
    return label.lower()


def extract_labels(metadata_path):
    metadata = read_json(metadata_path)  # labels is a list of strings
    return [normalize_label(tag['name']) for tag in metadata['tags']]


def collect_data(data_path='data'):
    for folder in glob(data_path + '/*'):
        if not os.path.isdir(folder):
            continue  # skip over non-directories

        ds_name = folder.split('/')[-1]
        labels_path = '{0}/{1}/{1}_labels.json'.format(data_path, ds_name)

        if os.path.isfile(labels_path):
            # load labels if file found
            labels = read_json(labels_path)  # labels is a list of strings
            # make sure loaded labels are normalized
            assert labels == [normalize_label(label) for label in labels]
        else:
            # extract labels if metadata file found, then write labels to file
            metadata_path = '{0}/{1}/{1}_metadata.json'.format(data_path, ds_name)
            labels = extract_labels(metadata_path)
            write_json(labels, labels_path)

        curr_dir = '{0}/{1}'.format(data_path, ds_name)
        for csv_path in glob(curr_dir + '/*.csv'):
            # table_name = '_'.join(csv_path.split('/')[-2:])
            dataset = pd.read_csv(csv_path, header=0)
            dataset = prep_dataset(dataset)  # TODO add prep_config, implement preprocessing
            yield csv_path, dataset, labels


def main(all_labels_path='data/labels.json', feat_map=rand_feat_map):
    X = []
    y = []
    headers = []
    all_labels = read_json(all_labels_path)
    label_to_index = {label: i for (i, label) in enumerate(all_labels)}

    for data_path, dataset, labels in collect_data():
        # TODO feature config, vocab/label config
        X.append(vectorize_dataset(dataset, feat_map))
        y.append(vectorize_labels(labels, label_to_index))
        headers.append(dataset.columns.values)  # TODO normalize headers

    # TODO save transformed data to file
    # print('data matrix:', X)
    # print('headers:', headers)
    # print('label vector:', y)


if __name__ == '__main__':
    main()
