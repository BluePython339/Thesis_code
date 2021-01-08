import tensorflow as tf

from tensorflow import keras
import random
from tqdm import tqdm

from tensorflow.keras.preprocessing.text import Tokenizer


class Decomp_tokenizer(object):

	def __init__(self):
		self.Tokenizer =Tokenizer(num_words =100, oov_token = "<NIIS>")

	def fit(self, data):
		#fdata = random.sample(data, 50)
		self.Tokenizer.fit_on_texts(data)

	def tokenizeData(self, data):
		return tf.convert_to_tensor(self.Tokenizer.texts_to_sequences(data))

	def __str__(self):
		return str(self.Tokenizer.word_index)