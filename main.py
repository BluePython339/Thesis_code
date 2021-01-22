import argparse
from tqdm import tqdm
import random
from MalMem import *
import tensorflow as tf
from preprocessor import Decomp_tokenizer
from sklearn.model_selection import train_test_split
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input csv with file names and types", type=str)
parser.add_argument("-d", "--data", help="Path to data files", type =str)

def map_files(csv_data, base_path):
	mapped_data = []
	print("[*Reading in data files*]")
	for i in tqdm(csv_data):
		#print(base_path+i[0])
		with open(base_path+i[0], 'r') as mfile:
			data = mfile.read().replace('\n', ' ')
			mapped_data.append((data, i[1]))
	return mapped_data

def split_train_test(full_data):
	y =  [i[1] for i in full_data]
	x =  [i[0] for i in full_data]
	x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.33, shuffle=True)

	return (x_train, y_train),(x_test, y_test)

def open_csv(fname):
	with open(fname, 'r') as f:
		data = f.readlines()

	csv_data = [i.strip('\n').split(',') for i in data]
	
	return csv_data

def compress_list(lists):
	fin = []
	for i in tqdm(lists):
		for a in i:
			fin.append(a)
	return fin

def np_list(lists):
	fin = []
	for i in tqdm(lists):
		fin.append(np.array(i))

	return np.array(fin)


if __name__ == "__main__":

	tokenizer = Decomp_tokenizer()
	args = parser.parse_args()
	#full_data = map_files(open_csv(args.input), args.data)

	#print(len(full_data))
	#train_data, test_data = split_train_test(full_data)
	#train_malware = train_data[0]
	#train_keys = train_data[1]
	#test_malware = test_data[0]
	#test_keys = test_data[1]

	#print("[*Fitting the tokenizer*]")
	#tokenizer.fit(tqdm(train_malware))
	#tokenizer.fit_label(train_keys)
	#tokenizer.save_status()

	#tokenizer.recover_status()
	#tokenizer.save_status()

	#print("[*Tokenizing data and writing to file*]")
	#tokenizer.tokenize_data_to_file("fitted_train_data", train_malware)
	#tokenizer.tokenize_data_to_file("fitted_test_data", test_malware)
	#tokenizer.tokenize_labels_to_file("fitted_train_keys", train_keys)
	#tokenizer.tokenize_labels_to_file("fitted_test_keys", test_keys)


	print("[*Testing the read back proccess*]")
	train_keys = tokenizer.read_data_from_file("fitted_train_keys")
	test_keys = tokenizer.read_data_from_file("fitted_test_keys")
	train_malware = tokenizer.read_data_from_file("fitted_train_data")
	test_malware = tokenizer.read_data_from_file("fitted_test_data")

	train_keys = compress_list(train_keys)

	train_malware = np_list(train_malware)
	train_keys = np.array(train_keys)
	print("[*All data ready for use*]")

	#print("[*Tokenising the test data*]")
	#test_malware = tokenizer.tokenizeData(test_malware)
	#print("[*Tokenising the train data*]")
	#train_malware = tokenizer.tokenizeData(train_malware)
	#train_key = train_data[1]
	#test_key = test_data[1]
	#train_key, key_map = tokenizer.tokenizeLabels(train_key)
	#test_key, _ = tokenizer.tokenizeLabels(test_key,key_map)

	#print(key_map)

	max_len = len(max(train_malware[:10], key=len))
	#model = MalMem(max_len, 64, 128, 12)
	#model.compile()
	print("[*Training the model*]")
	print("Train set: {}, Train key: {}".format(type(train_malware), type(train_keys)))
	print("Test set: {}, Test key: {}".format(len(test_malware), len(test_keys)))
	#model.fit(train_malware, train_keys, validation_data=(test_malware, test_keys) )
	train_rnn(train_malware[:10], train_keys[:10], max_len, 0)

	

