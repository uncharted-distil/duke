import itertools
import numpy as np

def parent_children_funcs(parent, children):
    def custom_agg(vector):
        if len(vector) > 1:
            return parent([vector[0],children(vector[1:])])
        else:
            return vector[0]
    return custom_agg

def build_combo_funcs():
    base_funcs = [np.mean, max, sum]
    return [parent_children_funcs(func_1, func_2) for (func_1, func_2) in itertools.permutations(base_funcs, 2)]

def combo_func_labels():
    base_funcs = ['np.mean', 'max', 'sum']
    return [func_1 + '+' + func_2 for (func_1, func_2) in itertools.permutations(base_funcs, 2)]