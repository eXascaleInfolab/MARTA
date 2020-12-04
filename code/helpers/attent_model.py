from keras.models import Model
from keras.layers import Input, RepeatVector, Permute, Dense, Embedding, Conv1D, MaxPooling1D, Dropout, \
    GlobalMaxPooling1D, Activation, LSTM, merge, Bidirectional, GRU, Flatten, Reshape
from keras.preprocessing.text import Tokenizer
from keras.layers.wrappers import TimeDistributed
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.initializers import Constant
from keras.layers import Dropout
import keras
import numpy as np

def define_model(nbr_filters,MAX_SEQUENCE_LENGTH,MAX_SENTENCE_NUM,grading_classes):
    print('Build model...')
    np.random.seed(1337)
    ############# input representation layer #######################
    inputs = Input(shape=(MAX_SENTENCE_NUM, MAX_SEQUENCE_LENGTH))
    ############ Sentence classification layer ##########################
    pi = TimeDistributed(Dense(grading_classes, activation='softmax'))(inputs)
    #We use separate LSTM modules to produce forward and backward hidden vectors, which are then concatenated:
    a = Bidirectional(LSTM(nbr_filters, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))(inputs)
    #to measure the importance of each sentence use tanh a one-layer MLP that produces an attention weight ai for the i-th
    # sentence,and Wa and ba are parameters in it
    a = TimeDistributed(Dense(1, activation='tanh'))(a)
    a = Flatten()(a) #A tensor, reshaped into 1-D
    #transform the result into a probability between 0 and 1 using the softmax function
    a = Activation('softmax',name="weights")(a)
    #Permutes the dimensions of the input according to a given pattern. e.g. (2, 1) permutes the first and second dimension of the input.
    pi = Permute((2, 1), name="pi")(pi)
    #we obtain a document-level distribution over class
    #labels as the weighted sum of sentence-level distributions:
    predictions = keras.layers.Dot(axes=(2, 1))([pi, a])
    _model = Model(inputs=inputs, outputs=predictions)
    _model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return _model

def retrain_model_weights (nbr_filters,MAX_SEQUENCE_LENGTH,MAX_SENTENCE_NUM,grading_classes):
    print('Build model...')
    np.random.seed(1337)
    ############# input representation layer #######################
    inputs = Input(shape=(MAX_SENTENCE_NUM, MAX_SEQUENCE_LENGTH))
    ############ Sentence classification layer ##########################
    pi = TimeDistributed(Dense(grading_classes, activation='softmax'))(inputs)
    #We use separate LSTM modules to produce forward and back- ward hidden vectors, which are then concatenated:
    a = Bidirectional(LSTM(nbr_filters, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))(inputs)
    #to measure the importance of each sentence use tanh a one-layer MLP that produces an attention weight ai for the i-th
    # sentence,and Wa and ba are parameters in it
    a = TimeDistributed(Dense(1, activation='tanh'))(a)
    a = Flatten()(a) #A tensor, reshaped into 1-D
    #transform the result into a probability between 0 and 1 using the softmax function
    weights = Activation('softmax',name="weights")(a)
    #Permutes the dimensions of the input according to a given pattern. e.g. (2, 1) permutes the first and second dimension of the input.
    pi = Permute((2, 1), name="pi")(pi)
    #we obtain a document-level distribution over classes
    #labels as the weighted sum of sentence-level distributions:
    predictions = keras.layers.Dot(axes=(2, 1), name="pred")([pi, weights])
    _model = Model(inputs=inputs, outputs=[predictions,weights])
    # opt = keras.optimizers.Adam(learning_rate=1e-5)
    _model.compile(loss={
        "pred": keras.losses.BinaryCrossentropy(from_logits=True),
        "weights": keras.losses.BinaryCrossentropy(from_logits=True),
    }, optimizer='rmsprop', metrics=['accuracy'])
    return _model

