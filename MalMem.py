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


def train_rnn(x_train, y_train, max_len, mask):
	epochs = 10
	batch_size = 200

	vec_dims = 1
	hidden_units = 256
	in_shape = (max_len, vec_dims)

	model = keras.Sequential()

	model.add(keras.layers.Masking(mask, name="in_layer", input_shape=in_shape,))
	model.add(keras.layers.LSTM(hidden_units, return_sequences=False))
	model.add(keras.layers.Dense(1, activation='relu', name='output'))

	model.compile(loss='binary_crossentropy', optimizer='rmsprop')
	print("well we got here at least")
	model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
	validation_split=0.05)

	return model
