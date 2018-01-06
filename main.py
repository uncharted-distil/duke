import numpy as np
import pandas as pd
import time
from gensim.models import Word2Vec
import re
import collections

# Measure distances to the curated list of types


def GetMostSimilarDBPediaType(test_column, unique_types, model, DEBUG):
    similarity_scores = np.zeros([len(test_column), len(unique_types)])
    start = time.time()
    exception_list = []
    for i in np.arange(len(unique_types)):
        for j in np.arange(len(test_column)):
            try:
                similarity_scores[j, i] = model.n_similarity(
                    test_column[j], unique_types[i])
                # print("DEBUG::similarity score: %f"%similarity_scores[j,i])
            except Exception as e:
                if e not in exception_list:
                    exception_list.append(e)
                    if(DEBUG):
                        print("DEBUG::SIMILARITY COMPUTATION EXCEPTION!!")
                        print(e)
                        print("DEBUG::type:")
                        print(unique_types[i])
                        # print("DEBUG::entity:")
                        # print(test_column[j])
    if(DEBUG):
        print("DEBUG::the required distance measure is:")
        print(similarity_scores)
        print("DEBUG::that took %f sec to compute!" % (time.time() - start))

    # Finally, measure the most similar class...
    most_similar_scores = np.amax(similarity_scores, axis=1)
    most_similar_indices = np.argmax(similarity_scores, axis=1)
    most_similar_types = np.array(unique_types)[most_similar_indices]

    # Display these results
    if(DEBUG):
        print("DEBUG::most similar scores (shape printed first)")
        print(most_similar_scores.shape)
        print(most_similar_scores)
        print("DEBUG::corresponding unique types:")
        print(most_similar_types)

    # Now, analyze in a bit more detail
    for i in np.arange(most_similar_types.shape[0]):
        most_similar_types[i] = ''.join(most_similar_types[i])
    most_similar_types_set = set(most_similar_types)
    most_similar_types_list = most_similar_types.tolist()
    if(DEBUG):
        print("DEBUG::corresponding unique types (parsed):")
        print(most_similar_types)
        print("DEBUG::corresponding unique types (nonredundant):")
        print(most_similar_types_set)
        print("DEBUG::frequencies of unique elements:")
        print(collections.Counter(most_similar_types_list))

    return most_similar_types_list, most_similar_scores, collections.Counter(most_similar_types_list), exception_list


# distance(w1, w2) Compute cosine distance between two words.
# distances(word_or_vector, other_words=()) Compute cosine distances from given word or vector to all words in other_words. If other_words is empty, return distance between word_or_vectors and all words in vocab.
# most_similar(positive=None, negative=None, topn=10, restrict_vocab=None, indexer=None) Find the top-N most similar words. Positive words contribute positively towards the similarity, negative words negatively.
# most_similar_cosmul(positive=None, negative=None, topn=10) Find the top-N most similar words, using the multiplicative combination objective proposed by Omer Levy and Yoav Goldberg in [4]. Positive words still contribute positively towards the similarity, negative words negatively, but with less susceptibility to one large distance dominating the calculation.
# most_similar_to_given(w1, word_list) Return the word from word_list most similar to w1.
# n_similarity(ws1, ws2) Compute cosine similarity between two sets of words.
# rank(w1, w2) Rank of the distance of w2 from w1, in relation to distances of all words from w1.
# similar_by_vector(vector, topn=10, restrict_vocab=None) Find the top-N most similar words by vector.
# similar_by_word(word, topn=10, restrict_vocab=None) Find the top-N most similar words.
# similarity(w1, w2) Compute cosine similarity between two words.
# wmdistance(document1, document2) Compute the Word Mover’s Distance between two documents. When using this code, please consider citing the following papers: Note that if one of the documents have no words that exist in the Word2Vec vocab, float(‘inf’) (i.e. infinity) will be returned. This method only works if pyemd is installed (can be installed via pip, but requires a C compiler).
# word_vec(word, use_norm=False) Accept a single word as input. Returns the word’s representations in vector space, as a 1D numpy array. If use_norm is True, returns the normalized word vector.
# words_closer_than(w1, w2) Returns all words that are closer to w1 than w2 is to w1.

def test(model):
    start = time.time()
    similarity = model.n_similarity(["Oprah", "Winfrey"], ["show", "television"])
    print("The similarity of [Oprah,Winfrey] with [show,television] is: ", similarity)
    print("similarity took %f seconds to measure" % (time.time() - start))
    
    start = time.time()
    similarity = model.n_similarity(["Oprah", "Winfrey"], ["Oprah", "Winfrey"])
    print("The similarity of [Oprah,Winfrey] with [Oprah, Winfrey] is: ", similarity)
    print("similarity took %f seconds to measure" % (time.time() - start))


def load_model(model_name='wiki2vec'):
    ''' Load a word2vec model from a file in models/ '''
    models = {
        'wiki2vec': 'en_1000_no_stem/en.model',
    }
    start = time.time()
    print("loading model...")
    model = Word2Vec.load("models/{0}".format(models[model_name]))
    print("model loaded in {0} seconds".format(time.time() - start))
    return model

def normalize_text(words, flatten=True):
    lol = [w.replace('_', ' ').lower().split(' ') for w in words]  # list of lists of single words
    if flatten:
        return [el for sublist in lol for el in sublist]
    else:
        return lol

def main(dataset_name = '185_baseball', n_keep = 20):
    
    model = load_model()

    # load types from local file
    with open('models/types', 'r') as f:  
        types = f.read().splitlines()
        
    types = [re.findall('[A-Z][^A-Z]*', t) for t in types]  # types is a lol with types split by capitalization
    print('types: ', types)
   
    csv_path = 'data/{0}/{0}_dataset/tables/learningData.csv'.format(dataset_name)
    full_df = pd.read_csv(csv_path, header=0)  # read csv assuming first line has header text
    text_df = full_df.select_dtypes(['object'])  # drop non-text rows (pandas strings are of type 'object')
    # TODO confirm that the columns selected can't be cast to a numeric type to avoid numeric strings (e.g. '1')

    # package data: concat then normalize headers and text columns
    all_headers = full_df.columns.values
    text = np.concatenate([text_df[h].values for h in text_df.columns])
    words = np.concatenate([all_headers, text])  
    words = normalize_text(words)
    
    # print('all headers: ', normalize_text(all_headers))

    distances = np.zeros(len(types))
    n_processed = 0
    for w in words:
        try:
            distances += model.wv.distances(w, types)
            n_processed += 1
        except KeyError as err:
            print('error checking distance of word {0} to types (out of vocab?):'.format(w), err)
    distances /= max(1, n_processed)  # divide to get average 

    # sort types by cumulative/average distance 
    sort_indices = np.argsort(distances)
    sorted_types = np.array(types)[sort_indices]

    print('top {0} types: '.format(n_keep), sorted_types[:n_keep])
    

    # # BEGIN TESTING!!
    # print("First, evaluate each element/cell in test column:")
    # most_similar_types_list,
    # most_similar_scores,
    # type_counts,
    # exception_list = GetMostSimilarDBPediaType(test_column,
    #                                            unique_types,
    #                                            model,
    #                                            DEBUG
    #                                            )
    # print("These are the various DBpedia types and their prediction frequencies:")
    # print(type_counts)
    # # print("These are the word exceptions encountered:")
    # # print(exception_list)
    # print("Second, evaluate test column cumulatively (after some filtering...):")
    # # but first, filter out trivial assertions while merging test_column into one long list
    # test_column_merged = []
    # for i in np.arange(len(test_column)):
    #     if(most_similar_scores[i] > 0.1):
    #         test_column_merged.extend(test_column[i])
    # tmp = test_column_merged
    # test_column_merged = []
    # test_column_merged.append(tmp)
    # if(DEBUG):
    #     print("DEBUG::these is the merged test column:")
    #     print(test_column_merged)
    # most_similar_types_list, similarity_scores, type_counts, exception_list = GetMostSimilarDBPediaType(
    #     test_column_merged, unique_types, model, DEBUG)
    # print("These are the various DBpedia types and their prediction frequencies:")
    # print(type_counts)
    # # print("These are the word exceptions encountered:")
    # # print(exception_list)


if __name__ == '__main__':
    main()
