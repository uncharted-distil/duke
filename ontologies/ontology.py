# import owlready2 as owl
import ontospy
import os
import json
from inflection import underscore

def to_class_name(class_obj):
    return underscore(str(class_obj.bestLabel())).replace('_', ' ') 

def has_relations(class_relations):
    return (len(class_relations['children']) > 0) or (len(class_relations['parents']) > 0)

def get_relationships_file_name(ontology_name, prune=True):
    return 'class-relationships_{0}{1}.json'.format(ontology_name, '_pruned' if prune else '')

def gen_class_relationships_file(ontology_name = 'dbpedia_2016-10', prune=True):
    print('loading ontology: ', ontology_name)
    ont = ontospy.Ontospy('{0}.nt'.format(ontology_name))

    print('building class relationships')
    relationships = {}
    for cl in ont.classes:
        relationships[to_class_name(cl)] = {
            'parents': [to_class_name(c) for c in cl.parents()], 
            'children': [to_class_name(c) for c in cl.children()], 
            }

    if prune:
        relationships = {name: rels for (name, rels) in relationships.items() if has_relations(rels)}

    print('writing class relationships to file')
    file_name = get_relationships_file_name(ontology_name, prune)
    with open(file_name, 'w') as f:
        json.dump(relationships, f, indent=4)


if __name__ == '__main__':
    gen_class_relationships_file()