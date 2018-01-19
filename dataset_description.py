import json
import time
import pandas as pd
from datetime import datetime
from inflection import underscore, pluralize

import numpy as np
from gensim.models import Word2Vec
from DatasetLoader import DatasetLoader
from SampleProcessor import SampleProcessor
# from identify_subject import getSentenceFromKeywords
from similarity_functions import w2v_similarity
# from utils import (get_timestamp, load_dataset, load_embedding, load_ontology,
#                    log_top_similarities, timeit, NumpyEncoder)

from trees import tree_score
from ontologies.ontology import get_tree_file_name


class DatasetDescriptor(object):

    def __init__(self, 
        dataset=None,
        embedding_path='./models/word2vec/en_1000_no_stem/en.model',  # wiki2vec model
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
        self.dataset_loader = DatasetLoader(embedding=self.embedding, vprint=self.vprint)
        self.tree = self.load_ontology(ontology_path)
        self.classes = list(self.tree.keys())


        # make multi-word classes into lists before handing to sim func
        classes_lol = [cl.split(' ') if isinstance(cl, str) else cl for cl in self.classes]  
        self.similarity_func = lambda words: similarity_func(words, classes_lol, self.embedding)
        self.source_agg_func = source_agg_func
        self.tree_agg_func = tree_agg_func

        self.sample_processor = SampleProcessor(similarity_func=self.similarity_func, classes=self.classes, vprint=self.vprint)

        if dataset:
            self.process_dataset(dataset)
    
    def get_description(self, dataset=None, reset_scores=False):

        if reset_scores:
            assert(dataset)  # if resetting scores, a new dataset should be provided
            self.sample_processor.reset_scores()

        if dataset:
            self.process_dataset(dataset)
        
        tree_scores = {src: self.aggregate_tree_scores(source=src) for src in self.sources()}
        final_scores = self.aggregate_source_scores(tree_scores)

        top_word = self.classes[np.argmax(final_scores)]
        description = 'This dataset is about {0}.'.format(pluralize(top_word))
        self.vprint('\n\n dataset description:', description, '\n\n')

        return(description)

    def process_dataset(self, dataset):
        data = self.dataset_loader.load_dataset(dataset)
        self.sim_scores, self.n_samples_seen = self.sample_processor.process_data(data, self.max_num_samples)


    def similarity_scores(self, source):
        return self.sim_scores[source] / max(1, self.n_samples_seen[source])


    def aggregate_tree_scores(self, scores=None, source=None):
        # check that one and only one of scores and source are provided
        if not scores and not source: raise Exception('must provide score or source')
        if scores and source: raise Exception('should only provide either score or source, not both')

        # get scores from source if source (and not scores) provided
        scores = scores if scores else self.similarity_scores(source)

        # convert score to dict that maps class to score if needed
        score_map = scores if isinstance(scores, dict) else dict(zip(self.classes, scores))

        agg_score_map = tree_score(score_map, self.tree, self.tree_agg_func)
        return np.array([agg_score_map[cl] for cl in self.classes])


    def sources(self):
        return list(self.sim_scores.keys())


    def aggregate_source_scores(self, scores):
        if isinstance(scores, dict):
            scores = list(scores.values())                
        assert(len(scores) == len(self.sources()))
        return self.source_agg_func(scores)

    


    def normalize_class_tree(self, tree):
        # filter out keys with out-of-vocab words -- all words in class name must be in vocab
        tree = {name: rels for (name, rels) in tree.items() if self.dataset_loader.in_vocab(name)}
        classes = list(tree.keys())  # filtered class list
        #     log('dropped {0} out of {1} type values for having out-of-vocab words. \n'.format()

        # remove filtered classes from parent and child lists
        for _ , rels in tree.items(): 
            rels['children'] = [cl for cl in rels['children'] if (cl in classes)] 
            rels['parents'] = [cl for cl in rels['parents'] if (cl in classes)] 

        return tree


    def load_embedding(self, embedding_path='./models/word2vec/en_1000_no_stem/en.model'):
        ''' Load a word2vec embedding from a file in embeddings/ '''
        self.vprint('loading word2vec embedding model')
        return Word2Vec.load(embedding_path)
    

    def load_ontology(self, ontology_path='dbpedia_2016-10', prune=True):
        self.vprint('loading class ontology:', ontology_path)
        tree_file_name = get_tree_file_name(ontology_path, prune)
        with open('ontologies/{0}'.format(tree_file_name), 'r') as f:  
            tree = json.load(f)

        return self.normalize_class_tree(tree)
