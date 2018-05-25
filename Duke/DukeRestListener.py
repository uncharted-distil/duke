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
import sys

from Duke.agg_functions import *
from Duke.dataset_descriptor import DatasetDescriptor
from Duke.utils import mean_of_rows

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

            dataset_path=fileName
            tree_path='ontologies/class-tree_dbpedia_2016-10.json'
            embedding_path='embeddings/wiki2vec/en.model'
            row_agg_func=mean_of_rows
            tree_agg_func=parent_children_funcs(np.mean, max)
            source_agg_func=mean_of_rows
            max_num_samples = 1e6
            verbose=True
            
            duke = DatasetDescriptor(
                    dataset=dataset_path,
                    tree=tree_path,
                    embedding=embedding_path,
                    row_agg_func=row_agg_func,
                    tree_agg_func=tree_agg_func,
                    source_agg_func=source_agg_func,
                    max_num_samples=max_num_samples,
                    verbose=verbose,
                    )

            print('initialized duke dataset descriptor \n')

            description = duke.get_dataset_description()
            print(description)

            N = 5
            out = duke.get_top_n_words(N)
            print("The top N=%d words are"%N)
            print(out)

            print("The whole script took %f seconds to execute"%(time.time()-start))

            return self.encoder.encode(out)


config = configparser.ConfigParser()
config.read('config.ini')
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
