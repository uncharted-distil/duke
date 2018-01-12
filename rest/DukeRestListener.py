from flask import Flask, request
from json import JSONEncoder
import pandas as pd
import numpy as np
import configparser
import os
import time
from gensim.models import Word2Vec
import re
import collections

class DukeRestListener:
	""" DukeRestListener accepts a .csv file, uses its predictive model to
	generate a summary for the file, then encodes it in JSON to be
	returned to the caller
	"""
	def __init__(self, modelName):
		self.modelName = modelName
		self.encoder = JSONEncoder()

	# Measure distances to the curated list of types
	def GetMostSimilarDBPediaType(self,test_column,unique_types,model,DEBUG):
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

	def predictFile(self, fileName, sim_threshold):
		## SET SOME MACRO PARAMETERS HERE
		HEADER = False # use the header for classification, rather than some column
		DEBUG = False # print debug information (warning -> very verbose!!) 
		STRING_NORMALIZATION = 'NONE' # normalize test strings to a) CAPS - all caps b) LOWER - all lower case
					# c) PROPER - capitalize first letter d) NONE - don't do anything
		MAX_CELLS = 50 # however many top rows to look at
	
		test_key = 'Position' # make sure header exists (should be a list of strings if several words)
		
		# Load the 2015 wiki2vec word2vec model
		start=time.time()
		print("loading wiki2vec model...")
		#print("DEBUG::")
		#print('/clusterfiles/en_1000_no_stem/en.model')
		#print(self.modelName)
		model = Word2Vec.load('/clusterfiles/en_1000_no_stem/en.model')
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

		# Load some text from the dataset/dataframe
		dataset_frame = pd.read_csv(str(fileName),header=0, nrows=MAX_CELLS, dtype='str')
	
		dataset_frame_headers = list(dataset_frame)
	
		if(HEADER):
			test_column = dataset_frame_headers # test the header, instead of some column
			print("Testing the header...")
		else:
			test_column = dataset_frame[test_key].values # test column as list
			print("Testing column "+test_key)
	
		if(DEBUG):
			print("DEBUG::header:")
			print(dataset_frame_headers)

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

		print("DEBUG::the test column is:")
		print(test_column)

		if(DEBUG):
			print("DEBUG::there are %d entries in the test column"%len(test_column))
			print("DEBUG::the test column is:")
			print(test_column)

		#### BEGIN TESTING!!
		print("First, evaluate each element/cell in test column:")
		most_similar_types_list, most_similar_scores, type_counts, exception_list = self.GetMostSimilarDBPediaType(test_column,unique_types,model,DEBUG)
		print("These are the various DBpedia types and their prediction frequencies:")
		print(type_counts)
		print("This is the test column:")
		print(test_column)
		print("These are predicted types (followed by scores):")
		print(most_similar_types_list)
		print(most_similar_scores)
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
		most_similar_types_list, most_similar_scores, type_counts, exception_list = self.GetMostSimilarDBPediaType(test_column_merged,unique_types,model,DEBUG)
		print("These are the various DBpedia types and their prediction frequencies:")
		print(type_counts)
		print("This is the test column:")
		print(test_column_merged)
		print("These are predicted types (followed by scores):")
		print(most_similar_types_list)
		print(most_similar_scores)
		#print("These are the word exceptions encountered:")
		#print(exception_list)
		print("The whole script took %f seconds to execute"%(time.time()-start))

		results = most_similar_types_list

		return self.encoder.encode(("This dataset is about "+results[0]))

config = configparser.ConfigParser()
config.read('rest/config.ini')
modelName = config['DEFAULT']['modelName']
        
listener = DukeRestListener(modelName)

app = Flask(__name__)
    
@app.route("/fileUpload", methods=['POST'])
def predictUploadedFile():

	""" Listen for data being POSTed on root.
	"""
	# print("DEBUG::predictUploadedFile::chkpt0")
	request.get_data()
	file = request.files['file']
	fileName = '/clusterfiles/uploaded_file.csv'
	file.save(fileName)
	sim_threshold=0.1 # this will need to move into the thin-client eventually, as a hyper-parameter
	# print("DEBUG::predictUploadedFile::chkpt1")	
	result = listener.predictFile(fileName,sim_threshold)
	# print("DEBUG::predictUploadedFile::chkpt2")
	os.remove(fileName)

	return result
