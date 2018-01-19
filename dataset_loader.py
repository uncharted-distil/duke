import numpy as np
import pandas as pd
from inflection import underscore, pluralize


class DatasetLoader:

    def __init__(self, embedding, verbose=False):
        self.vprint = print if verbose else lambda *a, **k: None
        self.embedding = embedding


    @staticmethod
    def normalize_words(words, to_list=True, replace_chars = {'_': ' ', '-': ' '}):
        words = underscore(words)  # converts to snake_case
        for old, new in replace_chars.items(): 
            words = words.replace(old, new)
        if to_list:
            return words.split(' ')
        else:
            return words


    @staticmethod
    def get_dropped(old, new):
        return set(old).difference(set(new))


    def in_vocab(self, word_list):
        if isinstance(word_list, str):
            word_list = word_list.split(' ')
        return all([word in self.embedding.wv.vocab for word in word_list])


    def remove_out_of_vocab(self, word_groups):
        if isinstance(word_groups, str):
            word_groups = word_groups.split(' ')
        
        if not isinstance(word_groups, np.ndarray):
            word_groups = np.array(word_groups)
        
        # removes all word lists with any oov words
        in_vocab = [self.in_vocab(group) for group in word_groups]
        self.vprint('dropping {0} out of {1} values for having out-of-vocab words \n'.format(len(word_groups) - sum(in_vocab), len(word_groups)))
        return word_groups[in_vocab]

    
    def format_data(self, data):
        word_groups = np.array([self.normalize_words(words) for words in data])  # list of lists of single words
        return self.remove_out_of_vocab(word_groups)


    def load_dataset(self, dataset, drop_nan=True):

        self.vprint('loading dataset')
        
        if isinstance(dataset, str):
            csv_path = 'data/{0}.csv'.format(dataset)
            dataset = pd.read_csv(csv_path, header=0)  # read csv assuming first line has header text. TODO handle files w/o headers
        else:
            assert isinstance(dataset, pd.DataFrame)

        headers = dataset.columns.values
        text_df = dataset.select_dtypes(['object'])  # drop non-text rows (pandas strings are of type 'object')
        # TODO confirm that the columns selected can't be cast to a numeric type to avoid numeric strings (e.g. '1')

        dtype_dropped = self.get_dropped(headers, text_df.columns.values)
        self.vprint('dropped non-text columns: {0} \n'.format(list(dtype_dropped)))

        if drop_nan: # drop columns if there are any missing values
            # TODO handle missing values w/o dropping whole column
            text_df = text_df.dropna(axis=1, how='any')
            nan_dropped = self.get_dropped(headers, text_df.columns.values)
            nan_dropped = nan_dropped.difference(dtype_dropped)
            self.vprint('dropped columns with missing values: {0} \n'.format(list(nan_dropped)))
        
        out_data = {}
        self.vprint('normalizing headers \n')
        out_data['headers'] = self.format_data(headers)

        for col in text_df.columns.values:
            self.vprint('normalizing column: {0}\n'.format(col))
            out_data[self.normalize_words(col, to_list=False)] = self.format_data(text_df[col].values) 

        return out_data


