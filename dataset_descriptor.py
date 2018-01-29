import numpy as np
from inspect import signature
from operator import itemgetter
from inflection import pluralize

from class_tree import EmbeddedClassTree, tree_score
from dataset import EmbeddedDataset
from embedding import Embedding
from utils import mean_of_rows, no_op


class DatasetDescriptor():

    def __init__(self, 
        dataset='data/185_baseball.csv',
        columns=None,
        tree='ontologies/class-tree_dbpedia_2016-10.json',
        embedding='models/wiki2vec/en.model',
        row_agg_func=mean_of_rows,
        tree_agg_func=np.mean,
        source_agg_func=mean_of_rows,
        max_num_samples=int(1e5),
        verbose=False,
        ):

        # print function that works only when verbose is true
        self.vprint = print if verbose else no_op
        self.max_num_samples = max_num_samples

        self.embedding = embedding if isinstance(embedding, Embedding) else Embedding(embedding_path=embedding, verbose=verbose)
        self.dataset = dataset if isinstance(dataset, EmbeddedDataset) else EmbeddedDataset(self.embedding, columns=columns, dataset_path=dataset, verbose=verbose)
        self.tree = tree if isinstance(tree, EmbeddedClassTree) else EmbeddedClassTree(self.embedding, tree_path=tree, verbose=verbose)

        self.row_agg_func = row_agg_func
        self.source_agg_func = source_agg_func
        self.tree_agg_func = tree_agg_func

        self.similarity_matrices = {}

    @property
    def classes(self):
        return self.tree.classes

    @property
    def sources(self):
        return list(self.similarity_matrices.keys())
    
    
    def compute_similarity_matrices(self, reset_matrices=True):

        class_matrix = self.tree.class_vectors.T

        # compute cosine similarity bt embedded data and ontology classes for each source
        for src, data_matrix in self.dataset.data_vectors.items():

            self.vprint('computing class similarity for data from:', src)
            
            sim_mat = np.dot(data_matrix, class_matrix)

            if reset_matrices or not self.similarity_matrices.get(src):
                self.similarity_matrices[src] = sim_mat
            else:
                self.similarity_matrices[src] = np.vstack([self.similarity_matrices[src], sim_mat])


    def get_dataset_class_scores(self):

        if not self.similarity_matrices:
            self.vprint('computing similarity matrices')
            self.compute_similarity_matrices()

        sources = self.sources
        self.stddev = np.mean([np.mean(np.std(self.similarity_matrices[src], axis=0)) for src in sources])

        # stacked_similarity = None
        # for src in sources:
        #     print(np.shape(stacked_similarity))
        #     print(np.shape(self.similarity_matrices[src]))
        #     if stacked_similarity is None:
        #         stacked_similarity = self.similarity_matrices[src]
        #     else:
        #         stacked_similarity = np.append(stacked_similarity, self.similarity_matrices[src], axis=1)
        # self.total_stddev = np.mean(np.std(stacked_similarity, axis=0))

        self.dataset.metadata['deviation_percentage'] = self.stddev / self.dataset.metadata['percent_unique']

        self.vprint('aggregating row scores')
        num_params = len(signature(self.row_agg_func).parameters)
        if num_params > 1:
            sim_scores = {src: self.row_agg_func(self.similarity_matrices[src], self.stddev) for src in sources}
        else:
            sim_scores = {src: self.row_agg_func(self.similarity_matrices[src]) for src in sources}

        
        self.vprint('aggregating tree scores')
        tree_scores = {src: self.aggregate_tree_scores(sim_scores[src]) for src in sources}
        
        self.vprint('aggregating source scores')
        return self.aggregate_source_scores(tree_scores), self.stddev
    
    def get_dataset_description(self):
        final_scores = self.get_dataset_class_scores()
        top_word = self.tree.classes[np.argmax(final_scores)]
        description = 'This dataset is about {0}.'.format(pluralize(top_word))
        self.vprint('\n\n dataset description:', description, '\n\n')

        return(description)
        

    def get_top_n_words(self, n):
        final_scores, stddev = self.get_dataset_class_scores()
        indexed_scores = zip(final_scores, range(len(final_scores)))
        indexed_scores = sorted(indexed_scores, key=itemgetter(0), reverse=True)
        top_n = indexed_scores[0:n]
        top_words = [self.tree.classes[index] for (score, index) in top_n]
        return top_words, stddev

    def aggregate_tree_scores(self, scores):
        # convert score to dict that maps class to score if needed
        score_map = scores if isinstance(scores, dict) else dict(zip(self.tree.classes, scores))
        
        # aggregate score over tree structure
        agg_score_map = tree_score(score_map, self.tree, self.tree_agg_func)
        
        # convert returned score map back to array
        return np.array([agg_score_map[cl] for cl in self.tree.classes]) 


    def aggregate_source_scores(self, scores):

        assert len(scores) == len(self.sources)
        if isinstance(scores, dict):
            scores = list(scores.values())                
        num_params = len(signature(self.source_agg_func).parameters)
        if num_params > 1:
            return self.source_agg_func(scores, self.stddev)
        else:
            return self.source_agg_func(scores)
