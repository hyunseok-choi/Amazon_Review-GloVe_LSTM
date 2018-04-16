
# coding: utf-8

# Dataset from: http://jmcauley.ucsd.edu/data/amazon/

# ### Data Loading

import pandas as pd
import gzip
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Embedding, Flatten, Dense, LSTM, Dropout, Input, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras import optimizers
from keras.utils.training_utils import multi_gpu_model
import os

def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)

def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

df = getDF('reviews_Electronics_5.json.gz')
df.loc[df['overall']>=4.0, 'label'] = 1
df.loc[df['overall']<4.0, 'label'] = 0

# Undersampling to deal with imbalanced data
neg_sample = df[df.label == 0]
pos_indices = df[df.label == 1].index
random_indices = np.random.choice(pos_indices, len(neg_sample), replace=False)
pos_sample = df.loc[random_indices]

df = pd.concat([pos_sample, neg_sample], ignore_index=True)
print("Data shape after undersample: ", df.shape)

# Data Tokenization
maxlen = 100 # Cut reviews after 100 words
max_words = 10000 # Only consider the top 10,000 most common words in the dataset
input_texts = np.array(df['reviewText'])

# TODO-Tokenization needs to be done after train/test set split
filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n0123456789'
tokenizer = Tokenizer(num_words=max_words, filters=filters)
tokenizer.fit_on_texts(input_texts) # This builds the word index
sequences = tokenizer.texts_to_sequences(input_texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen = maxlen)
labels = np.asarray(df['label'])
print('Shape of data tensor: ', data.shape)
print('Shape of label tensor:', labels.shape)

X_train, X_test, y_train, y_test = train_test_split(data, labels,
                                                    test_size=0.2, random_state=777)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                             test_size = 0.2, random_state = 777)

# Embedding Preprocessing

glove_dir = '../../kaggle/input/'

embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'), encoding='utf8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

embedding_dim = 100
embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if i < max_words:
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

# # Tuning Learning rate (the results didn't converge in the previous execution)
# adam = optimizers.Adam(lr=0.0005)

# Defining baseline model (with 70% test accuracy)
# print('Build model...')
# model = Sequential()
# model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
# model.add(Flatten())
# model.add(Dense(32, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

# # Load GloVe embeding
# model.layers[0].set_weights([embedding_matrix])
# model.layers[0].trainable = False
# print(model.summary())

# # Training and evaluation
# model.compile(optimizer=adam,
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
# history = model.fit(X_train, y_train,
#                     epochs=10,
#                     batch_size=32,
#                     validation_data=(X_val, y_val))
# print("Result: ", model.metrics_names, model.evaluate(X_test, y_test))

# LSTM
print('Build LSTM model...')
inp = Input(shape=(maxlen, ))
x = Embedding(max_words, embedding_dim, weights=[embedding_matrix])(inp)
x = Bidirectional(LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1,
    name='lstm_layer'))(x)
x = GlobalMaxPool1D()(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(1, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model = multi_gpu_model(model, gpus=4) # Use 4 GPUs in pararrel
model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
print(model.summary())                  
batch_size = 32
epochs = 20
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, 
        validation_data=(X_val, y_val), verbose=2)
# Testing LSTM
print("Test result: ", model.metrics_names, model.evaluate(X_test, y_test))