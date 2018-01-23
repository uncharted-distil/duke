import json
import time
import pandas as pd
from datetime import datetime
from inflection import underscore, pluralize

import numpy as np
from gensim.models import Word2Vec
from similarity_functions import w2v_similarity
from trees import tree_score
from ontologies.ontology import get_tree_file_name
from dataset_loader import DatasetLoader
from random import shuffle


class DatasetDescriptor():

    def __init__(self, 
        dataset=None,
        embedding_path='en_1000_no_stem/en.model',  # wiki2vec model
        ontology_path='dbpedia_2016-10',
        similarity_func=w2v_similarity,
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
        self.similarity_func = lambda words: similarity_func(words, classes_lol, self.embedding)
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


    def get_dataset_class_scores(self, dataset=None, reset_scores=False):

        if reset_scores:
            assert dataset is not None # if resetting scores, a new dataset should be provided
            self.reset_scores()

        if dataset:
            self.process_dataset(dataset)
        
        self.vprint('aggregating tree scores')
        tree_scores = {src: self.aggregate_tree_scores(source=src) for src in self.sources()}
        
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
        self.n_samples_seen = {}
        self.sim_scores = {}


    def process_dataset(self, dataset):

        data = self.dataset_loader.load_dataset(dataset)

        for source, text in data.items():
            self.process_samples(source, text)


    def process_samples(self, source, text):

        self.vprint('processing samples from:', source)
        
        if self.max_num_samples and len(text) > self.max_num_samples:
            self.vprint('subsampling word list of length {0} to {1}'.format(len(text), self.max_num_samples))
            shuffle(text)  # TODO problem to shuffle in place -- effects outside method? 
            text = text[:self.max_num_samples]

        for words in text:  # TODO vectorize
            try:
                if not source in self.sim_scores.keys():
                    self.sim_scores[source] = np.zeros(len(self.classes))
                
                if not source in self.n_samples_seen.keys():
                    self.n_samples_seen[source] = 0

                self.sim_scores[source] += self.similarity_func(words)
                self.n_samples_seen[source] += 1

            except KeyError as err:
                print('error checking distance of word {0} to classes (out of vocab?):'.format(words), err)
                raise err
            except Exception as err:
                print('unknown error: ', err)
                print('text being processed: {0}'.format(words))
                raise err


    def similarity_scores(self, source):
        return self.sim_scores[source] / max(1, self.n_samples_seen[source])


    def aggregate_tree_scores(self, scores=None, source=None):
        # check that one and only one of scores and source are provided
        if not scores and not source: raise Exception('must provide score or source')
        if scores and source: raise Exception('should only provide either score or source, not both')

        # get scores from source if source (and not scores) provided
        scores = scores if scores else self.similarity_scores(source)

        # convert score to dict that maps class to score if needed
        score_map = score_map if isinstance(scores, dict) else dict(zip(self.classes, scores))

        agg_score_map = tree_score(score_map, self.tree, self.tree_agg_func)
        return np.array([agg_score_map[cl] for cl in self.classes]) # convert returned score map back to array


    def sources(self):
        return list(self.sim_scores.keys())


    def aggregate_source_scores(self, scores):
        if isinstance(scores, dict):
            scores = list(scores.values())                
        assert(len(scores) == len(self.sources()))

        #return self.source_agg_func(scores)
        return np.mean(scores, axis=0)

    def normalize_class_tree(self, tree):
        # filter out keys with out-of-vocab words -- all words in class name must be in vocab
        tree = {name: rels for (name, rels) in tree.items() if self.in_vocab(name)}
        classes = list(tree.keys())  # filtered class list
        #     log('dropped {0} out of {1} type values for having out-of-vocab words. \n'.format()

        # remove filtered classes from parent and child lists
        for _ , rels in tree.items(): 
            rels['children'] = [cl for cl in rels['children'] if (cl in classes)] 
            rels['parents'] = [cl for cl in rels['parents'] if (cl in classes)] 

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
