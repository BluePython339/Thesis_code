import argparse
from tqdm import tqdm
import random
from MalMem import *
import tensorflow as tf
from preprocessor import Decomp_tokenizer
from sklearn.model_selection import train_test_split
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

def map_outputs(all, mapping=None):
	outs = []

	if not mapping:
		mapping = {}
		unique = list(set(all))
		for index, i in enumerate(unique):
			mapping[i] = index

	for i in all:
		outs.append(tf.convert_to_tensor(mapping[i]))

	return  outs, mapping


if __name__ == "__main__":
	key_map = {}
	#print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
	#exit()
	tokenizer = Decomp_tokenizer()
	args = parser.parse_args()
	full_data = map_files(open_csv(args.input), args.data)

	#print(len(full_data))
	train_data, test_data = split_train_test(full_data)
	train_malware = train_data[0]
	print("[*Fitting the tokenizer*]")
	tokenizer.fit(tqdm(train_malware))
	test_malware = test_data[0]
	print("[*Tokenising the test data*]")
	test_malware = tokenizer.tokenizeData(test_malware)
	print("[*Tokenising the train data*]")
	train_malware = tokenizer.tokenizeData(train_malware)
	train_key = train_data[1]
	test_key = test_data[1]
	train_key, key_map = map_outputs(train_key)
	test_key, _ = map_outputs(test_key,key_map)
	print(key_map)

	model = MalMem(1000, 64, 128, 12)
	model.compile()
	print("[*Training the model*]")
	model.fit(train_malware, train_key, validation_data=(test_malware, test_key))

	

