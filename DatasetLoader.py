import numpy as np
import pandas as pd
from inflection import underscore, pluralize

class DatasetLoader:
    def __init__(self, embedding=None, vprint=print):
        self.vprint = vprint
        self.embedding = embedding

    def load_dataset(self, csv_path, drop_nan=True):

        self.vprint('loading dataset')
        
        if isinstance(csv_path, str):
            dataset = pd.read_csv(csv_path, header=0)  # read csv assuming first line has header text. TODO handle files w/o headers
        else: 
            assert(isinstance(dataset, pd.DataFrame))
        
        headers = dataset.columns.values

        # TODO confirm that the columns selected can't be cast to a numeric type to avoid numeric strings (e.g. '1')
        text_df = dataset.select_dtypes(['object'])  # drop non-text rows (pandas strings are of type 'object')
        # dtype_dropped = get_dropped(headers, text_df.columns.values)
        # self.vprint('dropped non-text columns: {0} \n'.format(list(dtype_dropped)), verbose)

        if drop_nan: # drop columns if there are any missing values
            # TODO handle missing values w/o dropping whole column
            text_df = text_df.dropna(axis=1, how='any')
            # nan_dropped = get_dropped(headers, text_df.columns.values)
            # nan_dropped = nan_dropped.difference(dtype_dropped)
            # self.vprint('dropped columns with missing values: {0} \n'.format(list(nan_dropped)), verbose)
        
        out_data = {}
        self.vprint('normalizing headers \n')
        out_data['headers'] = self.format_data(headers)

        for col in text_df.columns.values:
            self.vprint('normalizing column: {0}\n'.format(col))
            out_data[self.normalize_words(col, to_list=False)] = self.format_data(text_df[col].values) 

        return out_data

    def format_data(self, data):
        word_groups = np.array([self.normalize_words(words) for words in data])  # list of lists of single words
        return self.remove_out_of_vocab(word_groups)

    @staticmethod
    def normalize_words(words, to_list=True, replace_chars = {'_': ' ', '-': ' '}):
        words = underscore(words)  # converts to snake_case
        for old, new in replace_chars.items(): 
            words = words.replace(old, new)
        if to_list:
            return words.split(' ')
        else:
            return words


    def remove_out_of_vocab(self, word_groups):
        if isinstance(word_groups, str):
            word_groups = word_groups.split(' ')
        
        if not isinstance(word_groups, np.ndarray):
            word_groups = np.array(word_groups)
        
        # removes all word lists with any oov words
        in_vocab = [self.embedding.in_vocab(group) for group in word_groups]
        self.vprint('dropping {0} out of {1} values for having out-of-vocab words \n'.format(len(word_groups) - sum(in_vocab), len(word_groups)))
        return word_groups[in_vocab]