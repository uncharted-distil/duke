# import owlready2 as owl
import ontospy
import os
import json
from inflection import underscore


# TODO merge with normalize_text in dataset_loader
def to_class_name(class_obj):
    return underscore(str(class_obj.bestLabel())).replace('_', ' ').replace('-', ' ').replace('(', '').replace(')', '')


def has_relations(class_relations):
    return (len(class_relations['children']) > 0) or (len(class_relations['parents']) > 0)


def get_tree_file_name(ontology_name, prune=True):
    return 'class-relationships_{0}{1}.json'.format(ontology_name, '_pruned' if prune else '')


def generate_class_tree_file(ontology_name = 'dbpedia_2016-10', prune=False):
    print('loading ontology: ', ontology_name)
    ont = ontospy.Ontospy('{0}.nt'.format(ontology_name))

    all_classes = set()
    print('building class relationships')
    relationships = {}
    for cl in ont.classes:
        relationships[to_class_name(cl)] = {
            'parents': [to_class_name(p) for p in cl.parents()], 
            'children': [to_class_name(c) for c in cl.children()], 
            }

        parents = set([to_class_name(p) for p in cl.parents()])
        children = set([to_class_name(c) for c in cl.children()])
        all_classes = all_classes.union(parents, children, set([to_class_name(cl)]))

    print('pre prune relationships length:', len(relationships.keys()))

    if prune:
        # remove all isolated classes (dont have children or parents)
        relationships = {name: rels for (name, rels) in relationships.items() if has_relations(rels)}
    print('length of ontology classes:', len(ont.classes))
    print('length of all classes:', len(all_classes))
    print('length of (post-prune) relationships keys:', len(relationships.keys()))
    print('classes minus rel keys:', all_classes.difference(set(relationships.keys())))
    # assert(len(all_classes) == len(relationships.keys()))

    print('writing class relationships to file')
    file_name = get_tree_file_name(ontology_name, prune)
    with open(file_name, 'w') as f:
        json.dump(relationships, f, indent=4)


if __name__ == '__main__':
    generate_class_tree_file()