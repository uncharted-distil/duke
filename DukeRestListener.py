#!/usr/bin/python3
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

import cProfile as profile


from dataset_description import DatasetDescriptor
from similarity_functions import w2v_similarity


class DukeRestListener:
	""" DukeRestListener accepts a .csv file, uses its predictive model to
	generate a summary for the file, then encodes it in JSON to be
	returned to the caller
	"""
	def __init__(self, modelName):
		self.modelName = modelName
		self.encoder = JSONEncoder()

	# Describe a given dataset
	def predictFile(self, fileName, sim_threshold):

		start = time.time()		

		dataset=fileName
    		embedding_path='en_1000_no_stem/en.model'  # wiki2vec model
    		ontology_path='dbpedia_2016-10'
    		similarity_func=w2v_similarity
    		tree_agg_func=np.mean
    		source_agg_func=lambda scores: np.mean(scores, axis=0),
    		max_num_samples = 2000
    		verbose=True

    		duke = DatasetDescriptor(
        		# dataset=dataset,
        		dataset=None,
        		embedding_path=embedding_path,
        		ontology_path=ontology_path,
        		similarity_func=similarity_func,
        		tree_agg_func=tree_agg_func,
        		source_agg_func=source_agg_func,
        		max_num_samples=max_num_samples,
        		verbose=verbose,
        		)

    		print('initialized duke dataset descriptor \n')

		description = duke.get_description(dataset)

		print(description)
		
		print("The whole script took %f seconds to execute"%(time.time()-start))
		

		return self.encoder.encode(description)


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
	sim_threshold=0.1 #may  move into the thin-client eventually, as hyper-parameter
	# print("DEBUG::predictUploadedFile::chkpt1")	
	result = listener.predictFile(fileName,sim_threshold)
	# print("DEBUG::predictUploadedFile::chkpt2")
	os.remove(fileName)

	return result
