import argparse
from tqdm import tqdm
import random
from MalMem import *
import tensorflow as tf
import json
from sizeChecker import get_file_size
from preprocessor import *
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input csv with file names and types", type=str)
parser.add_argument("-d", "--data", help="Path to data files", type =str)

def map_files(csv_data):
	mapped_data = []
	print("[*Reading in data files*]")
	for i in tqdm(csv_data):
		#print(base_path+i[0])
		with open(i[0], 'r') as mfile:
			data = mfile.read()
			mapped_data.append((data, i[1], i[0]))
	return mapped_data

def split_train_test(full_data):
	y =  [(i[1], i[2]) for i in full_data]
	x =  [i[0] for i in full_data]
	x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.33, shuffle=True)

	return (x_train, y_train),(x_test, y_test)

def open_csv(fname, base_path):
	fin = []
	with open(fname, 'r') as f:
		data = f.readlines()

	csv_data = [i.strip('\n').split(',') for i in data]
	for i in csv_data:
		if 100 < get_file_size(base_path+i[0]) < 3000000:
			fin.append((base_path+i[0],i[1]))
	return fin

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

def to_config(filename, train_len, test_len, max_len):
	a ={
	"train_len": train_len,
	"test_len": test_len,
	"max_len": max_len
	}
	with open(filename, 'w+') as f:
		f.write(json.dumps(a))

def split_to_file(filename, data):
	with open(filename, 'w+') as f:
		for i in data:
			f.write("{}\n".format(i))

if __name__ == "__main__":

	tokenizer = Decomp_tokenizer()
	args = parser.parse_args()
	full_data = map_files(open_csv(args.input,args.data))

	#print(len(full_data))
	train_data, test_data = split_train_test(full_data)
	#tokenizer.recover_status()

	train_malware = train_data[0]
	train_keys = [i[0] for i in train_data[1]]
	test_malware = test_data[0]
	test_keys = [i[0] for i in test_data[1]]

	train_files = [i[1] for i in train_data[1]]
	test_files = [i[1] for i in test_data[1]]

	split_to_file("test_set.csv", test_files)
	split_to_file("train_set.csv", train_files)


	print("[*Fitting the tokenizer*]")
	tokenizer.fit_args(tqdm(train_malware))
	tokenizer.fit_instr(tqdm(train_malware))
	tokenizer.fit_label(train_keys)
	#tokenizer.save_status()

	#tokenizer.recover_status()
	#tokenizer.save_status()

	print("[*Tokenizing data and writing to file*]")
	train_malware_args = tokenizer.tokenize_args(train_malware)
	train_malware_instr = tokenizer.tokenize_instr(train_malware)
	max_len_args = len(max(train_malware_args, key=len))
	max_len_inst = len(max(train_malware_instr, key=len))
	train_keys = tokenizer.tokenizeLabels(tqdm(train_keys))
	train_data_args = zip(train_malware_args, train_keys)
	train_data_instr = zip(train_malware_instr, train_keys)

	print("[*Train data fitted and ready for writing to file*]")
	test_malware_args = tokenizer.tokenize_args(test_malware)
	test_malware_instr = tokenizer.tokenize_instr(test_malware)
	test_keys = tokenizer.tokenizeLabels(tqdm(test_keys))
	test_data_args = zip(test_malware_args, test_keys)
	test_data_instr = zip(test_malware_instr, test_keys)

	train_len = len(train_malware_args)
	test_len = len(test_malware_args)

	to_config("tokenized_with_args.config", train_len,test_len,max_len_args)
	to_config("tokenized_with_instructions.config", train_len, test_len,max_len_inst)

	print("[* ARG TRAIN DATA TO FILE *]")
	for index, i in enumerate(tqdm(train_data_args)):
			write_to_data_file("args",i,str(index),test=False,max_len=max_len_args)
	print("[* ARG TEST DATA TO FILE *]")
	for index, i in enumerate(tqdm(test_data_args)):
		write_to_data_file("args",i,str(index), test=True,max_len=max_len_args)

	print("[* INSTR TRAIN DATA TO FILE *]")
	for index, i in enumerate(tqdm(train_data_instr)):
			write_to_data_file("instr",i,str(index),test=False,max_len=max_len_inst)
	print("[* INSTR TEST DATA TO FILE *]")
	for index, i in enumerate(tqdm(test_data_instr)):
		write_to_data_file("instr",i,str(index), test=True,max_len=max_len_inst)










