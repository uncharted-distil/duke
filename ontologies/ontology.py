# import owlready2 as owl
import ontospy
import os
import json
from inflection import underscore

def to_class_name(class_obj):
    return underscore(str(class_obj.bestLabel())).replace('_', ' ') 

def main(ontology_name = 'dbpedia_2016-10'):
    print('loading ontology: ', ontology_name)
    ont = ontospy.Ontospy('{0}.nt'.format(ontology_name))

    print('building class relationships')
    relationships = {}
    for cl in ont.classes:
        relationships[to_class_name(cl)] = {
            'parents': [to_class_name(c) for c in cl.parents()], 
            'children': [to_class_name(c) for c in cl.children()], 
            }

    print('writing class relationships to file')
    with open('class-relationships_{0}.json'.format(ontology_name), 'w') as f:
        json.dump(relationships, f, indent=2)


if __name__ == '__main__':
    main()