import tensorflow as tensorflow
from tensorflow import keras


def MalMem(vocab_size, emb_size,net_size, output_pars):
	model = keras.Sequential([
		keras.layers.Embedding(vocab_size, emb_size),
		keras.layers.Bidirectional(keras.layers.LSTM(net_size)),
		keras.layers.Dense(emb_size/2, activation='relu'),
		keras.layers.Dense(output_pars, activation='sigmoid')
		])
	return model

