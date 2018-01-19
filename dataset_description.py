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

        self.reset_scores()

        if dataset:
            self.process_dataset(dataset)


    def reset_scores(self):
        # dictionaries with data source as keys 
        self.n_samples_seen = {}
        self.sim_scores = {}


    def process_dataset(self, dataset):
        data = self.load_dataset(dataset)

        for source, text in data.items():
            self.process_samples(source, text)


    def process_samples(self, source, text):

        self.vprint('processing samples from:', source)
        
        if self.max_num_samples and len(text) > self.max_num_samples:
            self.vprint('subsampling word list of length {0} to {1}'.format(len(text), self.max_num_samples))
            shuffle(text)  # TODO problem to shuffle in place -- effects outside method? 
            text = text[:max_num_samples]

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
        return self.source_agg_func(scores)

    def get_dataset_class_scores(self, dataset=None, reset_scores=False):

        if reset_scores:
            assert(dataset)  # if resetting scores, a new dataset should be provided
            self.reset_scores()

        if dataset:
            self.process_dataset(dataset)
        
        tree_scores = {src: self.aggregate_tree_scores(source=src) for src in self.sources()}
        return self.aggregate_source_scores(tree_scores)

    
    def get_description(self, dataset=None, reset_scores=False):
        final_scores = self.get_dataset_class_scores(dataset, reset_scores)
        top_word = self.classes[np.argmax(final_scores)]
        description = 'This dataset is about {0}.'.format(pluralize(top_word))
        self.vprint('\n\n dataset description:', description, '\n\n')

        return(description)
    

    @staticmethod
    def normalize_words(words, to_list=True, replace_chars = {'_': ' ', '-': ' '}):
        words = underscore(words)  # converts to snake_case
        for old, new in replace_chars.items(): 
            words = words.replace(old, new)
        if to_list:
            return words.split(' ')
        else:
            return words


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


    def load_dataset(self, dataset, drop_nan=True):

        self.vprint('loading dataset')
        
        if isinstance(dataset, str):
            csv_path = 'data/{0}.csv'.format(dataset)
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






        # data = load_dataset(full_df, self.embedding, verbose=self.verbose)
        # similarities = self.run_trial(data, sim_func, extra_args)


    # def get_similarity_score(words, **kwargs):
    # similarities = {}
    #     for source, words in data.items():
    #         if(self.verbose):
    #             print('computing type similarity for ', source)
    #         similarities[source] = get_class_similarities(words, self.types, self.embedding, similarity_func, extra_args)

    

    # def run_trial(self, data, similarity_func=w2v_similarity, extra_args=None):

    #     self.vprint('running trial with similarity function: {0}{1}\n'.format(similarity_func.__name__, ', and extra args: {0}'.format(extra_args) if  extra_args else ''))
        
    #     similarities = {}
    #     for source, words in data.items():
    #         if(self.verbose):
    #             print('computing type similarity for ', source)
    #         similarities[source] = get_class_similarities(words, self.types, self.embedding, similarity_func, extra_args)

    #     return similarities
    
    # def produceSentenceFromDataframe(self, full_df):
            # similarities = self.run_trial(data, sim_func, extra_args)
            
            # write results of trial to file along with trial config info
            # record = {
            #     'embedding': self.embedding_name, 
            #     'types': self.types, 
            #     'sim_func': sim_func.__name__, 
            #     'extra_args': extra_args,
            #     'similarities': similarities,  # dict mapping column headers to similarity vectors (in the same order as types list)
            #     }
            
            # with open('trials/trial{0}'.format(get_timestamp()), 'w') as f:
            #     json.dump(record, f, cls=NumpyEncoder)

        # top_n = {}
        # n = 40
        # parsedTypes = [''.join(a) for a in self.types]
        # for header in similarities.keys():
        #     best = sorted(zip(parsedTypes, similarities[header]), key=lambda x: x[1], reverse=True)[0:10]
        #     top_n[header] = best
        #     # print(header)
        #     if(self.verbose):
        #         print(best)
        #         print(header + ": " + getSentenceFromKeywords(best, type_heirarchy_filename=self.type_heirarchy_filename))

        # all_types = []
        # for key in top_n.keys():
        #     all_types.extend(top_n[key])
        # all_types_aggregated = {}
        # for t_score in all_types:
        #     all_types_aggregated[t_score[0]] = 0.0
        # for t_score in all_types:
        #     all_types_aggregated[t_score[0]] = all_types_aggregated[t_score[0]] + t_score[1] 

        # all_types = []
        # for key in all_types_aggregated.keys():
        #     all_types.append((key, all_types_aggregated[key]))
        
        # return getSentenceFromKeywords(all_types)

# if __name__ == '__main__':
#     main()