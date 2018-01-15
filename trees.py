import numpy as np


def get_leaves(tree):
    return {name: rels for (name, rels) in tree.items() if not rels.get('children')} 


def aggregate_score(score_map, tree, agg_func):
    
    agg_score = {}
    processed = set()
    
    def all_children_aggd(node):
       return all([agg_score.get(ch) for ch in tree[node]['children']])

    def process_layer(layer):
        assert(layer)
        for node in layer:
            agg_score[node] = apply_agg_func(node, score_map, tree, agg_score, agg_func)
            # agg_score[node] = apply_agg_func(node_score, relations, agg_score, agg_func)
            processed.add(node)

    all_nodes = set(tree.keys())
    layer = get_leaves(tree).keys()

    # print('aggregating score for {0} nodes'.format(len(all_nodes)))
    # print('processing {0} leaves'.format(len(layer)))

    process_layer(layer)
    
    # while there are still unprocessed nodes, move up heirarchy
    while all_nodes.difference(processed):
        # print('moving up tree heirarchy')
        # print('number of nodes left:', len(all_nodes.difference(processed)))

        layer = set.union(*[set(tree[node]['parents']) for node in layer])  # get parents of previous layers
        layer = layer.difference(processed)  # remove already processed
        layer = [n for n in layer if all_children_aggd(n)]  # keep nodes where all child values have been computed
        process_layer(layer)
    
    # print('done aggregating over the tree \n')
    
    return agg_score


def apply_agg_func(node, score_map, tree, agg_score, agg_func=np.mean):
    score_list = [score_map[node]]
    relations = tree[node]
    if relations.get('children'):
        assert(all([agg_score.get(ch) for ch in relations['children']]))  # make sure all children have agg_scores already
        score_list = score_list + [agg_score[ch] for ch in relations['children']]

    return agg_func(score_list)  