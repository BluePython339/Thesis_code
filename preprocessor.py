import tensorflow as tf
import json
from tensorflow import keras
import random
from tqdm import tqdm
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np


class Decomp_tokenizer(object):

	def __init__(self):
		self.Tokenizer =Tokenizer(num_words=100, oov_token = "<NIIS>")
		self.label_mapping = {}

	def fit(self, data):
		#fdata = random.sample(data, 50)
		self.Tokenizer.fit_on_texts(data)

	def fit_label(self, data):
		mapping = {}
		unique = list(set(data))
		for index, i in enumerate(unique):
			mapping[i] = index
		self.label_mapping = mapping

	def tokenizeLabels(self, all):
		outs = []
		for i in all:
			outs.append(self.label_mapping.get(i))
		return outs

	def tokenize_labels_to_file(self, filename, data):
		outs = []
		ddict = {}
		for i in data:
			outs.append([self.label_mapping.get(i)])
		ddict["data"] = outs

		with open(filename+".json", "+w") as f:
			f.write(json.dumps(ddict))

	def read_data_from_file(self,filename):
		with open(filename+".json", 'r') as f:
			jdata = json.load(f)
			keys = jdata['data']
		print("[*Reading in {}.json*]".format(filename))
		return keys

	def tokenize_data_to_file(self, filename, data):
		tokens = self.Tokenizer.texts_to_sequences(tqdm(data))
		data = {
		}
		print("writing to dict")
		data["data"] = tokens
		print("writing dataset into file: {}.json".format(filename))
		with open(filename+".json", '+w') as f:
			f.write(json.dumps(data))

	def tokenizeData(self, data):
		tokens = self.Tokenizer.texts_to_sequences(tqdm(data))
		return tokens

	def __str__(self):
		return str(self.Tokenizer.word_index)

	def save_status(self,):
		with open("Tokenizer_data.json", '+w') as f:
			f.write(json.dumps(self.Tokenizer.to_json()))
		with open("Label_data.json", '+w') as f:
			f.write(json.dumps(self.label_mapping))

	def recover_status(self):
		with open("Label_data.json", 'r') as f:
			self.label_mapping = json.load(f)
		with open("Tokenizer_data.json", 'r') as f:
			self.Tokenizer = keras.preprocessing.text.tokenizer_from_json(json.load(f))


def write_to_data_file(data,filename, test=False, max_len=0):

	a = {
		"data": np_array_to_int(pad_sequences([data[0]], padding='post',maxlen=max_len)[0]),
		"key": data[1]
	}
	if test:
		with open("data/test/{}.json".format(filename),'w+') as f:
			f.write(json.dumps(a))
	else:
		with open("data/train/{}.json".format(filename),'w+') as f:
			f.write(json.dumps(a))

def np_array_to_int(data):
	fin = []
	for i in data:
		fin.append(int(i))
	return fin
