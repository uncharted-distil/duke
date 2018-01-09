import numpy as np


def w2v_similarity(words, types, model):
    return np.array([model.wv.n_similarity(words, typ) for typ in types])


def freq_nearest_similarity(words, types, model, extra_args={'n_nearest': 3}):
    n_nearest = extra_args['n_nearest']

    similarities = w2v_similarity(words, types, model)
    sorted_inds = np.argsort(similarities)[::-1]

    # return indicator vector for types that are among the n_nearest most similar
    neighbors = np.zeros(len(types))
    neighbors[sorted_inds[:n_nearest]] = 1  

    return neighbors


def get_type_similarities(data, types, model, similarity_func=w2v_similarity, extra_args=None):

    similarities = np.zeros(len(types))
    n_processed = 0
    for dat in data:
        try:
            if not np.all([d in model.wv.vocab for d in dat]):
                # skip with logging if not in vocab (should have been prevented by normalization)
                print('out of vocab: ', dat, [d in model.wv.vocab for d in dat])
            else:
                if extra_args:
                    similarities += similarity_func(dat, types, model, extra_args)
                else:
                    similarities += similarity_func(dat, types, model) 
                n_processed += 1
                
        except KeyError as err:
            print('error checking distance of word {0} to types (out of vocab?):'.format(dat), err)
            raise err
        except Exception as err:
            print('unknown error: ', err)
            raise err

    similarities /= max(1, n_processed)  # divide to get average 
    print('max, min average similarities: ', np.max(similarities), ', ', np.min(similarities), '\n\n')

    # sort types by average similarity and unpack lists 
    sort_indices = np.argsort(similarities)[::-1]
    sorted_types = np.array(types)[sort_indices]
    sorted_similarities = similarities[sort_indices]
    sorted_types = np.array([' '.join(typ) for typ in sorted_types])  # unpack lol with spaces between words and convert to np array

    return sorted_types, sorted_similarities