import sys
import numpy as np
import pandas as pd
from inflection import underscore
from Duke.utils import no_op, normalize_text, get_dropped, unit_norm_rows
from Duke.embedding import Embedding


class EmbeddedDataset:

    def __init__(self, embedding_model, dataset_path='data/185_baseball.csv', columns=None, max_num_samples=1e5, embed_dataset=True, verbose=False):
        
        self.data_vectors = {}
        self.max_num_samples=max_num_samples
        
        self.vprint = print if verbose else no_op

        assert isinstance(embedding_model, Embedding)
        self.embedding = embedding_model

        # set self.data by loading the file at the path given
        self.load_dataset(dataset_path, columns=columns)  
        
        if embed_dataset:
            self.embed_dataset()
        

    def format_data(self, data):
        word_groups = np.array([normalize_text(text) for text in data])  # list of lists of single words
        return self.embedding.remove_out_of_vocab(word_groups)


    def load_dataset(self, dataset, columns=None, drop_nan=True, reset_data=True):
        self.vprint('loading dataset {0}'.format(dataset if isinstance(dataset, str) else 'from pandas DataFrame'))
        
        if isinstance(dataset, str):
            dataset = pd.read_csv(dataset, header=0)  # read csv assuming first line has header text. TODO handle files w/o headers
        else:
            assert isinstance(dataset, pd.DataFrame)

        headers = dataset.columns.values
        if columns:
            text_df = dataset[columns]
        else:
            text_df = dataset.select_dtypes(['object'])  # drop non-text rows (pandas strings are of type 'object')
        # TODO confirm that the columns selected can't be cast to a numeric type to avoid numeric strings (e.g. '1')

        dtype_dropped = get_dropped(headers, text_df.columns.values)
        self.vprint('\ndropped non-text columns: {0}'.format(list(dtype_dropped)))

        if drop_nan: # drop columns if there are any missing values
            # TODO handle missing values w/o dropping whole column
            text_df = text_df.dropna(axis=1, how='any')
            nan_dropped = get_dropped(headers, text_df.columns.values)
            nan_dropped = nan_dropped.difference(dtype_dropped)
            if nan_dropped:
                self.vprint('\ndropped columns with missing values: {0}'.format(list(nan_dropped)))
        
        if not reset_data:
            # TODO implement variant where data is appended instead of overwritten
            raise Exception('not implemented')
            
        self.data = {}
        self.vprint('\nnormalizing headers')
        self.data['headers'] = self.format_data(headers)

        for col in text_df.columns.values:
            self.vprint('\nnormalizing column: {0}'.format(normalize_text(col, to_list=False)))
            self.data[normalize_text(col, to_list=False)] = self.format_data(text_df[col].values) 

        return self.data


    def embed_dataset(self, data=None, reset_scores=False):
        self.vprint('embedding dataset')
        
        data = data if data else self.data

        # compute data embedding for all sources in data
        for src, word_lol in data.items():
            self.vprint('computing data embedding for data from:', src)
            try:
                if self.max_num_samples and len(word_lol) > self.max_num_samples:
                    self.vprint('subsampling rows from length {0} to {1}'.format(len(word_lol), self.max_num_samples))
                    np.random.shuffle(word_lol)  # TODO minibatches rather than truncate / subsample?
                    word_lol = word_lol[:self.max_num_samples]

                dat_vecs = np.array([self.embedding.embed_multi_words(words) for words in word_lol])  # matrix of w/ len(data[src]) rows and n_emb_dim columns
                if len(dat_vecs) == 0:
                    continue
                dat_vecs = unit_norm_rows(dat_vecs)
            except:
                print(sys.exc_info())
                continue

            if reset_scores or not self.data_vectors.get(src):
                self.data_vectors[src] = dat_vecs
            else:
                self.data_vectors[src] = np.vstack([self.data_vectors[src], dat_vecs])
