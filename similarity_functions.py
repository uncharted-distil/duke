import numpy as np


def w2v_similarity(words, classes, model):
    return np.array([model.wv.n_similarity(words, cl) for cl in classes])/2 + 1/2  # normalize similarity between 0 and 1


def freq_nearest_similarity(words, classes, model, extra_args={'n_nearest': 3}):
    n_nearest = extra_args['n_nearest']

    similarities = w2v_similarity(words, classes, model)
    sorted_inds = np.argsort(similarities)[::-1]

    # return indicator vector for the n_nearest most similar classes
    neighbors = np.zeros(len(classes))
    neighbors[sorted_inds[:n_nearest]] = 1

    return neighbors


def get_class_similarities(data, classes, model, similarity_func=w2v_similarity, extra_args=None):

    # convert classes that are strings into lists of words
    classes = [cl.split(' ') if isinstance(cl, str) else cl for cl in classes]

    similarities = np.zeros(len(classes))
    n_processed = 0
    for dat in data:
        try:
            similarities += similarity_func(dat, classes, model, extra_args) if extra_args \
                       else similarity_func(dat, classes, model) 
            n_processed += 1
                
        except KeyError as err:
            print('error checking distance of word {0} to classes (out of vocab?):'.format(dat), err)
            raise err
        except Exception as err:
            print('unknown error: ', err)
            print('data being processed: {0}'.format(dat))
            raise err

    similarities /= max(1, n_processed)  # divide to get average 
    
    return similarities