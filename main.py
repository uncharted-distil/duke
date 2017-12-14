# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 21:59:58 2017

@author: azunre

Main script, which will use the wiki2vec 2015 model to try to summarize text entries
of a dataset (potentially using the simon semantic classifier as well...)
"""
import numpy as np
import pandas as pd
import time
from gensim.models import Word2Vec
import re
import collections

# Measure distances to the curated list of types
def GetMostSimilarDBPediaType(test_column,unique_types,model,DEBUG):
	similarity_scores = np.zeros([len(test_column),len(unique_types)])
	start = time.time()
	exception_list = []
	for i in np.arange(len(unique_types)):
		for j in np.arange(len(test_column)):
			try:
				similarity_scores[j,i] = model.n_similarity(test_column[j],unique_types[i])	
				#print("DEBUG::similarity score: %f"%similarity_scores[j,i])
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
		print("DEBUG::that took %f sec to compute!"%(time.time()-start))

	# Finally, measure the most similar class...
	most_similar_scores = np.amax(similarity_scores,axis=1)
	most_similar_indices = np.argmax(similarity_scores,axis=1)
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

#### EXAMPLE SCRIPT SHOWING HOW THE FUNCTION CAN BE USED...

## SET SOME MACRO PARAMETERS HERE
HEADER = False # use the header for classification, rather than some column
DEBUG = False # print debug information (warning -> very verbose!!) 
STRING_NORMALIZATION = 'NONE' # normalize test strings to a) CAPS - all caps
				# b) LOWER - all lower case c) PROPER - capitalize first letter d) NONE - don't do anything
MAX_CELLS = 50 # however many top rows to look at
test_key = 'Position' # make sure header exists (should be a list of strings if several words)
test_file_name = '185_baseball'

# Load the 2015 wiki2vec word2vec model
start=time.time()
print("loading model...")
model = Word2Vec.load("en_1000_no_stem/en.model")
if(DEBUG):
	print("DEBUG::2015 dBpedia model loaded in %f seconds"%(time.time()-start))

# Perform a test calculation to make sure things are working as expected
if(DEBUG):
	print("DEBUG::(UNIT TEST)::The similarity of [Oprah,Winfrey] with [show,television] is:")
	start = time.time()
	print(model.n_similarity(["Oprah","Winfrey"],["show","television"]))
	print("similarity took %f seconds to measure"%(time.time()-start))

# Now, load the unique types from file (as extracted by main_extracted_unique_types.py)
unique_types = []
with open('unique_types','r') as file: # this extracts lines
	unique_types = file.read().splitlines()
for i in np.arange(len(unique_types)): # this splits lines at capital letters, if several terms...
	unique_types[i] = re.findall('[A-Z][^A-Z]*',unique_types[i])
if(DEBUG):
	print("DEBUG::The are %d unique types"%len(unique_types))
	print("DEBUG::these unique types are:")
	print(unique_types)

# Load some text from a model dataset
dataset_frame = pd.read_csv("/vectorizationdata/KnowledgeGraph2Vec/wiki2vec/darpa_data/seeds/"+test_file_name+"/"+test_file_name+"_dataset/tables/learningData.csv",header=0)
test_column = dataset_frame[test_key].values # test column as list
dataset_frame_headers = list(dataset_frame)

if(DEBUG):
	print("DEBUG::header:")
	print(dataset_frame_headers)

if(HEADER):
	test_column = dataset_frame_headers # test the header, instead of some column

for i in np.arange(len(test_column)): # this removes underscores -> _ and creates word list
	test_column[i] = test_column[i].rsplit('_',-1)
	if(STRING_NORMALIZATION=='CAPS'):
		for j in np.arange(len(test_column[i])):
			test_column[i][j]=test_column[i][j].upper()
	elif(STRING_NORMALIZATION=='LOWER'):
		for j in np.arange(len(test_column[i])):
			test_column[i][j]=test_column[i][j].lower()
	elif(STRING_NORMALIZATION=='PROPER'):
		for j in np.arange(len(test_column[i])):
			test_column[i][j]=test_column[i][j].title()
		

test_column = test_column[0:MAX_CELLS] #only look at a "manageable" subset (for human analysis), unless header

if(DEBUG):
	print("DEBUG::there are %d entries in the test column"%len(test_column))
	print("DEBUG::the test column is:")
	print(test_column)

#### BEGIN TESTING!!
print("First, evaluate each element/cell in test column:")
most_similar_types_list, most_similar_scores, type_counts, exception_list = GetMostSimilarDBPediaType(test_column,unique_types,model,DEBUG)
print("These are the various DBpedia types and their prediction frequencies:")
print(type_counts)
#print("These are the word exceptions encountered:")
#print(exception_list)
print("Second, evaluate test column cumulatively (after some filtering...):")
test_column_merged = [] #but first, filter out trivial assertions while merging test_column into one long list
for i in np.arange(len(test_column)):
	if(most_similar_scores[i]>0.1):
		test_column_merged.extend(test_column[i])
tmp = test_column_merged
test_column_merged = []
test_column_merged.append(tmp)
if(DEBUG):
	print("DEBUG::these is the merged test column:")
	print(test_column_merged)
most_similar_types_list, similarity_scores, type_counts, exception_list = GetMostSimilarDBPediaType(test_column_merged,unique_types,model,DEBUG)
print("These are the various DBpedia types and their prediction frequencies:")
print(type_counts)
#print("These are the word exceptions encountered:")
#print(exception_list)
