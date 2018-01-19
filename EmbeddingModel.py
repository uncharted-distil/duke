from gensim.models import Word2Vec

class EmbeddingModel:
    def __init__(self, embedding_path='./models/word2vec/en_1000_no_stem/en.model'):
        ''' Load a word2vec embedding from a file in embeddings/ '''
        self.model = Word2Vec.load(embedding_path)

    def n_similarity(self, words, classes):
        return self.model.wv.n_similarity(words, classes)

    def in_vocab(self, word_list):
        if isinstance(word_list, str):
            word_list = word_list.split(' ')
        return all([word in self.model.wv.vocab for word in word_list])

     