import numpy as np

from utils import timeit, load_types, load_dataset, load_model, log_top_similarities
from similarity_functions import w2v_similarity, freq_nearest_similarity, get_type_similarities


def trial(headers, text, types, model, similarity_func=w2v_similarity, extra_args=None, use_headers=True, use_text=True):

    if use_headers and use_text:
        data = np.concatenate([headers, text]) # TODO allow equal-weighted header and text input, rather than proportional to num samples / headers
    elif use_headers:
        data = headers # TODO add other functions for combining headers and text
    elif use_text:
        data = text
    else: 
        raise Exception('at least one of use_headers and use_text must be true')

    sorted_types, similarities = get_type_similarities(data, types, model, similarity_func, extra_args)
    # sorted_types, similarities = timeit(get_type_similarities, [data, types, model, similarity_func, extra_args])

    return sorted_types, similarities

# config = sim_func, extra_args, use_header/text, dataset, (types)
def main(
    dataset_name='185_baseball', 
    configs = [
        {'similarity_function': w2v_similarity},
        {'similarity_function': freq_nearest_similarity},
        # {'similarity_function': freq_nearest_similarity, 'extra_args': {'n_nearest': 3}},
    ]):
    
    print('loading model')
    model = load_model()
    # model = timeit(load_model)
    
    print('loading types')
    types = load_types(model)
    
    print('loading dataset')
    headers, text = load_dataset(dataset_name, model)

    results = {}
    for conf in configs:
        sim_func = conf['similarity_function']
        extra_args = conf.get('extra_args')

        print('running trial with similarity function: ', sim_func.__name__, '\n')
        if extra_args:
            print('with extra args: ', extra_args, '\n')

        print('headers only: \n')
        sorted_types, similarities = trial(headers, text, types, model, sim_func, extra_args, use_headers=True, use_text=False)
        results['{0}-headers'.format(sim_func.__name__)] = {'types': types, 'similarities': similarities}
        log_top_similarities(sorted_types, similarities)

        print('text only: \n')
        sorted_types, similarities = trial(headers, text, types, model, sim_func, extra_args, use_headers=False, use_text=True)
        results['{0}-text'.format(sim_func.__name__)] = {'types': types, 'similarities': similarities}
        log_top_similarities(sorted_types, similarities)
        
        print('both headers and text: \n')
        sorted_types, similarities = trial(headers, text, types, model, sim_func, extra_args, use_headers=True, use_text=True)
        results['{0}-both'.format(sim_func.__name__)] = {'types': types, 'similarities': similarities}
        log_top_similarities(sorted_types, similarities)

    # print(results)
    return results


if __name__ == '__main__':
    main()

