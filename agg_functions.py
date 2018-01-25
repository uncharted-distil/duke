import itertools
from utils import max_of_rows, mean_of_rows
import numpy as np

def null_prep(vector):
    return vector

def exponential(vector):
    return [np.exp(v) for v in vector]

def step(vector):
    return [v if v > 0.5 else v*0.1 for v in vector]

def quadratic(vector):
    return [v * v for v in vector]

def sqrt(vector):
    return [np.sqrt(v) for v in vector]

def parent_children_funcs(parent, children, prep=null_prep):
    def custom_agg(vector):
        vector = prep(vector)
        if len(vector) > 1:
            return parent([vector[0],children(vector[1:])])
        else:
            return vector[0]
    return custom_agg

def build_combo_funcs(prep=null_prep):
    base_funcs = [np.mean, max]
    return [parent_children_funcs(func_1, func_2, prep=prep) for (func_1, func_2) in itertools.permutations(base_funcs, 2)]

def combo_func_labels(prefix=''):
    base_funcs = ['np.mean', 'max']
    return [prefix + func_1 + '+' + func_2 for (func_1, func_2) in itertools.permutations(base_funcs, 2)]

def build_threshold_mean_max(threshold):
    def threshold_row_agg(vector, val):
        if(val < threshold):
            return max_of_rows(vector)
        else:
            return mean_of_rows(vector)
    return threshold_row_agg
