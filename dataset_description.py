import json
import itertools
from random import shuffle

import numpy as np
from gensim.models import Word2Vec
from inflection import pluralize

from dataset_loader import DatasetLoader
from ontologies.ontology import get_tree_file_name
from similarity_functions import w2v_similarity
from trees import tree_score



class DatasetDescriptor():

    def __init__(self, 
        dataset=None,
        embedding_path='en_1000_no_stem/en.model',  # wiki2vec model
        ontology_path='dbpedia_2016-10',
        similarity_func=w2v_similarity,
        row_agg_func=lambda scores: np.mean(scores, axis=0),
        tree_agg_func=np.mean,
        source_agg_func=lambda scores: np.mean(scores, axis=0),
        max_num_samples=None,
        verbose=False,
        ):
    
        # print function that works only when obj is init to verbose
        self.vprint = print if verbose else lambda *a, **k: None
        self.max_num_samples = max_num_samples

        # load embedding before ontology as embedding is used to remove out of vocab words from the ontology        
        self.embedding = self.load_embedding(embedding_path)
        self.tree = self.load_ontology(ontology_path)
        self.classes = list(self.tree.keys())

        # make multi-word classes into lists before handing to sim func
        classes_lol = [cl.split(' ') if isinstance(cl, str) else cl for cl in self.classes]  
        self.class_vectors = np.array([self.mean_embedding(words) for words in classes_lol])
        self.class_vectors = self.unit_norm_rows(self.class_vectors)

        self.similarity_func = lambda words: similarity_func(words, classes_lol, self.embedding)
        self.row_agg_func = row_agg_func
        self.source_agg_func = source_agg_func
        self.tree_agg_func = tree_agg_func


        self.dataset_loader = DatasetLoader(self.embedding, verbose)
        self.reset_scores()  

        if dataset:
            self.process_dataset(dataset)

    # TODO move this to separate embedding model object so its not copied here and in dataset (could call from self.dataset_loader, but seems like the wrong place)
    def in_vocab(self, word_list):
        if isinstance(word_list, str):
            word_list = word_list.split(' ')
        return all([word in self.embedding.wv.vocab for word in word_list])


    def mean_embedding(self, word_list):
        return np.mean([self.embedding.wv[word] for word in word_list], axis=0)


    @staticmethod
    def unit_norm_rows(vectors):
        return vectors/np.linalg.norm(vectors, axis=1)[:,None]


    def get_dataset_class_scores(self, dataset=None, reset_scores=False):

        if dataset:
            self.process_dataset(dataset, reset_scores)

        sources = self.sources()

        self.vprint('aggregating row scores')
        sim_scores = {src: self.row_agg_func(self.similarity_matrices[src]) for src in sources}
        
        self.vprint('aggregating tree scores')
        tree_scores = {src: self.aggregate_tree_scores(sim_scores[src]) for src in sources}
        
        self.vprint('aggregating source scores')
        return self.aggregate_source_scores(tree_scores)

    
    def get_description(self, dataset=None, reset_scores=False):
        final_scores = self.get_dataset_class_scores(dataset, reset_scores)
        top_word = self.classes[np.argmax(final_scores)]
        description = 'This dataset is about {0}.'.format(pluralize(top_word))
        self.vprint('\n\n dataset description:', description, '\n\n')

        return(description)


    def reset_scores(self):
        # dictionaries with data source as keys 
        self.data_vectors = {}
        self.similarity_matrices = {}


    def process_dataset(self, dataset, reset_scores=False):

        self.vprint('processing dataset')

        data = self.dataset_loader.load_dataset(dataset)

        # compute cosine similarity bt embedded data and ontology classes
        for src, word_lol in data.items():
            self.vprint('computing class similarity for data from:', src)

            if self.max_num_samples and len(word_lol) > self.max_num_samples:
                self.vprint('subsampling rows from length {0} to {1}'.format(len(word_lol), self.max_num_samples))
                np.random.shuffle(word_lol)  # TODO minibatches rather than truncate / subsample?
                word_lol = word_lol[:self.max_num_samples]

            dat_vecs = np.array([self.mean_embedding(words) for words in word_lol])  # matrix of w/ len(data[src]) rows and n_emb_dim columns
            dat_vecs = self.unit_norm_rows(dat_vecs)
            sim_mat = np.dot(dat_vecs, self.class_vectors.T)

            if reset_scores or not self.data_vectors.get(src):
                self.data_vectors[src] = dat_vecs
            else:
                self.data_vectors[src] = np.vstack([self.data_vectors[src], dat_vecs])

            if reset_scores or not self.similarity_matrices.get(src):
                self.similarity_matrices[src] = sim_mat
            else:
                self.similarity_matrices[src] = np.vstack([self.similarity_matrices[src], sim_mat])


    def aggregate_tree_scores(self, scores):
        # convert score to dict that maps class to score if needed
        score_map = scores if isinstance(scores, dict) else dict(zip(self.classes, scores))
        
        # aggregate score over tree structure
        agg_score_map = tree_score(score_map, self.tree, self.tree_agg_func)
        
        # convert returned score map back to array
        return np.array([agg_score_map[cl] for cl in self.classes]) 


    def sources(self):
        return list(self.similarity_matrices.keys())


    def aggregate_source_scores(self, scores):
        if isinstance(scores, dict):
            scores = list(scores.values())                
        assert len(scores) == len(self.sources())
        return self.source_agg_func(scores)


    def normalize_class_tree(self, tree):
        # filter out keys with out-of-vocab words -- all words of the class name must be in vocab
        tree = {name: rels for (name, rels) in tree.items() if self.in_vocab(name)}
        classes = list(tree.keys())  # filtered class list

        # remove filtered classes from parent and child lists
        for cl, rels in tree.items(): 
            tree[cl]['children'] = [child for child in rels['children'] if (child in classes)] 
            tree[cl]['parents'] = [parent for parent in rels['parents'] if (parent in classes)] 

        return tree


    def load_embedding(self, embedding_path='en_1000_no_stem/en.model'):
        ''' Load a word2vec embedding from a file in embeddings/ '''
        self.vprint('loading word2vec embedding model')
        return Word2Vec.load('embeddings/{0}'.format(embedding_path))
    

    def load_ontology(self, ontology_path='dbpedia_2016-10', prune=False):
        self.vprint('loading class ontology:', ontology_path)
        tree_file_name = get_tree_file_name(ontology_path, prune)
        with open('ontologies/{0}'.format(tree_file_name), 'r') as f:  
            tree = json.load(f)

        return self.normalize_class_tree(tree)
    