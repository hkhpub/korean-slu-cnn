"""
copyright Kwangho Heo
2016-12-06
"""
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten, merge
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout
from keras.models import Model
from keras.callbacks import EarlyStopping

import numpy as np

VALIDATION_SPLIT = 0.2
MAX_SEQUENCE_LENGTH = 207
EMBEDDING_DIM = 100

nb_filter = 200
filter_sizes = [2, 3, 4]


class SLU_SA_CNN:

    def __init__(self, texts, labels, labels_index, prev_labels):
        self.texts = texts
        self.labels_index = labels_index

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)

        word_index = tokenizer.word_index
        for prev_label in set(prev_labels):
            word_index[prev_label] = len(word_index)
        print 'Found %s unique tokens.' % len(word_index)

        data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
        labels = to_categorical(np.asarray(labels))
        print 'Shape of data tensor:', data.shape
        print 'Shape of label tensor:', labels.shape

        for i, feature_x in enumerate(data):
            feature_x[0] = word_index[prev_labels[i]]

        self.tokenizer = tokenizer
        self.data = data
        self.labels = labels
        self.word_index = word_index

    def train(self):

        # maybe cross validation is needed

        indices = np.arange(self.data.shape[0])
        np.random.shuffle(indices)
        data = self.data[indices]
        labels = self.labels[indices]
        nb_validation_samples = int(VALIDATION_SPLIT * self.data.shape[0])

        x_train = data[:-nb_validation_samples]
        y_train = labels[:-nb_validation_samples]
        x_val = data[-nb_validation_samples:]
        y_val = labels[-nb_validation_samples:]

        embedding_layer = Embedding(len(self.word_index) + 1,
                                    EMBEDDING_DIM,
                                    input_length=MAX_SEQUENCE_LENGTH)

        sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)

        conv_0 = Conv1D(nb_filter, filter_sizes[0], activation='relu')(embedded_sequences)
        maxpool_0 = MaxPooling1D(MAX_SEQUENCE_LENGTH - filter_sizes[0] + 1)(conv_0)
        # conv_0 = Conv1D(nb_filter, filter_sizes[0], activation='relu')(maxpool_0)
        # maxpool_0 = MaxPooling1D(35)(conv_0)  # global max pooling

        conv_1 = Conv1D(nb_filter, filter_sizes[1], activation='relu')(embedded_sequences)
        maxpool_1 = MaxPooling1D(MAX_SEQUENCE_LENGTH - filter_sizes[1] + 1)(conv_1)
        # conv_1 = Conv1D(nb_filter, filter_sizes[1], activation='relu')(maxpool_1)
        # maxpool_1 = MaxPooling1D(35)(conv_1)  # global max pooling

        conv_2 = Conv1D(nb_filter, filter_sizes[2], activation='relu')(embedded_sequences)
        maxpool_2 = MaxPooling1D(MAX_SEQUENCE_LENGTH - filter_sizes[2] + 1)(conv_2)
        # conv_2 = Conv1D(nb_filter, filter_sizes[2], activation='relu')(maxpool_2)
        # maxpool_2 = MaxPooling1D(35)(conv_2)  # global max pooling

        merged_tensor = merge([maxpool_0, maxpool_1, maxpool_2], mode='concat', concat_axis=1)
        flatten = Flatten()(merged_tensor)
        dropout = Dropout(0.2)(flatten)
        preds = Dense(labels.shape[1], activation='softmax')(dropout)

        model = Model(sequence_input, preds)
        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['acc'])

        earlyStopping = EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')
        # happy learning!
        model.fit(x_train, y_train, validation_data=(x_val, y_val),
                  nb_epoch=100, batch_size=50,
                  callbacks=[earlyStopping])
        pass
