from gensim.models import Word2Vec
from Duke.utils import mean_of_rows, no_op
import numpy as np


class Embedding:
    def __init__(self, 
        embedding_path='embeddings/wiki2vec.model',
        embed_agg_func= mean_of_rows,
        verbose=False):
        ''' Load a word2vec embedding from a file '''
        
        self.vprint = print if verbose else no_op
        self.embed_agg_func = embed_agg_func

        self.vprint('loading word2vec embedding model')    
        self.model = Word2Vec.load(embedding_path)


    def remove_out_of_vocab(self, word_groups):
        if isinstance(word_groups, str):
            word_groups = word_groups.split(' ')
        
        if not isinstance(word_groups, np.ndarray):
            word_groups = np.array(word_groups)
        
        # removes all word lists with any oov words
        in_vocab = [self.in_vocab(group) for group in word_groups]
        self.vprint('dropping {0} out of {1} values for having out-of-vocab words'.format(len(word_groups) - sum(in_vocab), len(word_groups)))
        return word_groups[in_vocab]


    def embed_multi_words(self, word_list):
        return self.embed_agg_func([self.model.wv[word] for word in word_list])


    def n_similarity(self, words, classes):
        return self.model.wv.n_similarity(words, classes)


    def in_vocab(self, word_list):
        if isinstance(word_list, str):
            word_list = word_list.split(' ')
        return all([word in self.model.wv.vocab for word in word_list])

     
