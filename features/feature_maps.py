import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer, CountVectorizer
from sklearn.feature_extraction import FeatureHasher
from itertools import combinations
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np


class IIDFeatures(object):
    def transform(self, D):
        '''
        For each column d of D, transform the string from each cell of the column into a vector. Then average the
        vectors across cells (because we assume each cell in a datasets column is independent of all other cells).
        Then return a sparse csr matrix where each row is the average feature vector from a column of D.

        :param D: Dataset array that has shape (n_rows, n_cols) where each element is a string
        :return: X, the transformed feature vector for each column of D, of shape (n_cols, n_features)
        '''
        rows = []
        for d in D.T:
            rows.append(self.feat_map.transform(d).mean(axis=0).A1)
        return sp.csr_matrix(rows)
        # return sp.vstack([sp.csr_matrix(self.feat_map.transform(d).mean(axis=0).A1) for d in D.T])


class NGramProduct(IIDFeatures):

    def __init__(self, ngram_range, n_features, binary=False):
        '''
        uses a character-level ngram count vectorizer and hashes down to n_features, then adds the product of those features (indices also hashed) to the feature array

        :param ngram_range: tuple (min_n, max_n) for size of ngrams used for features
        :param n_features: dimension of vector output by transform(D).
        '''
        # analyzer='char_wb', mean v. l1 / l2 norm, ngram + poly degree + hash dimensions
        self.base_map = HashingVectorizer(analyzer='char', ngram_range=ngram_range, n_features=n_features)
        # TODO compare to keras.preprocessing.text.hashing_trick
        # TODO compare to sparse polynomial pacakage / sklearn 0.2
        self.feat_hash = FeatureHasher(n_features=n_features)
        # FeatureHasher(n_features=1048576, input_type=’dict’, dtype=<class ‘numpy.float64’>, alternate_sign=True, non_negative=False)

    @staticmethod
    def pair_combos(X, row):
        ind_slice = slice(X.indptr[row], X.indptr[row+1])
        pairs = zip(X.indices[ind_slice], X.data[ind_slice])
        return combinations(pairs, 2)

    def transform_column(self, d):
        X = self.base_map.transform(d)
        Y = self.feat_hash.transform({
            '{0},{1}'.format(ind0, ind1): val0*val1
            for (ind0, val0), (ind1, val1) in self.pair_combos(X, row)}
            for row in range(X.shape[0]))

        return X + Y

        # print('len X:', len(X.data))
        # print('len Y:', len(Y.data))
        # print('combined len:', len(X.data) + len(Y.data))
        # print('len X+Y:', len(Z.data))


class HashedNGramFeatures(IIDFeatures):
    def __init__(self, ngram_range, n_features):
        '''
        uses a character-level ngram count vectorizer and hashes down to n_features

        :param ngram_range: tuple (min_n, max_n) for size of ngrams used for features
        :param n_features: dimension of vector output by transform(D).
        '''
        self.feat_map = HashingVectorizer(analyzer='char', ngram_range=ngram_range, n_features=n_features)


class FullNGramFeatures(IIDFeatures):
    def __init__(self, ngram_range, max_features=None):
        '''
        uses a character-level ngram count vectorizer

        :param ngram_range: tuple (min_n, max_n) for size of ngrams used for features
        :param max_features: chooses the most frequent ngrams if max_features is set (not None)
        '''
        self.feat_map = CountVectorizer(analyzer=u'char', ngram_range=ngram_range, max_features=max_features)

    def fit_transform(self, D):
        self.feat_map.fit(D.flatten())
        return self.transform(D)


if __name__ == '__main__':
    ngp = NGramProduct((1, 3), int(1e5))
    df = pd.read_csv('data/test.csv')
    d = df[df.columns[0]]
    print(ngp.transform_column(d))
