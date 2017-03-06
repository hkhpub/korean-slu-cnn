"""
copyright Kwangho Heo
2016-12-06
"""
from keras.layers import Dense, Input, Flatten, merge
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout
from keras.models import Model

import numpy as np
from keras.callbacks import EarlyStopping

EMBEDDING_DIM = 100
DROPOUT_RATE = 0.5
BATCH_SIZE = 60

NUM_EPOCHS = 20

NUM_FILTERS = 200
filter_sizes = [3, 4, 5]

EARLY_STOPPING = True
# EARLY_STOPPING = False

MODEL_FILE_PATH = '../models/keras_cnn_text_model.h5'


class SLU_SA_CNN:

    model = None

    def __init__(self, word_index, x_train, y_train, x_val, y_val, sequence_length):
        self.word_index = word_index
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.sequence_length = sequence_length

    def define_model_topology(self):
        embedding_layer = Embedding(len(self.word_index) + 1,
                                    EMBEDDING_DIM,
                                    input_length=self.sequence_length)

        sequence_input = Input(shape=(self.sequence_length,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)

        conv_0 = Conv1D(NUM_FILTERS, filter_sizes[0], activation='relu')(embedded_sequences)
        maxpool_0 = MaxPooling1D(self.sequence_length - filter_sizes[0] + 1)(conv_0)

        conv_1 = Conv1D(NUM_FILTERS, filter_sizes[1], activation='relu')(embedded_sequences)
        maxpool_1 = MaxPooling1D(self.sequence_length - filter_sizes[1] + 1)(conv_1)

        conv_2 = Conv1D(NUM_FILTERS, filter_sizes[2], activation='relu')(embedded_sequences)
        maxpool_2 = MaxPooling1D(self.sequence_length - filter_sizes[2] + 1)(conv_2)

        merged_tensor = merge([maxpool_0, maxpool_1, maxpool_2], mode='concat', concat_axis=1)
        flatten = Flatten()(merged_tensor)

        dropout = Dropout(DROPOUT_RATE)(flatten)
        preds = Dense(self.y_train.shape[1], activation='softmax')(dropout)

        model = Model(sequence_input, preds)
        model.compile(loss='categorical_crossentropy',
                      optimizer='rmsprop',
                      metrics=['acc'])
        return model

    def train(self):
        print 'start training...'
        self.model = self.define_model_topology()

        if EARLY_STOPPING:
            earlyStopping = EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')
            # happy learning!
            self.model.fit(self.x_train, self.y_train, validation_data=(self.x_val, self.y_val),
                           nb_epoch=NUM_EPOCHS, batch_size=BATCH_SIZE,
                           callbacks=[earlyStopping])
        else:
            self.model.fit(self.x_train, self.y_train, validation_data=(self.x_val, self.y_val),
                           nb_epoch=NUM_EPOCHS, batch_size=BATCH_SIZE)
        # save model
        self.model.save_weights(MODEL_FILE_PATH)
        pass

    def evaluate(self):
        print 'start evaluating...'
        self.model = self.define_model_topology()
        self.model.load_weights(MODEL_FILE_PATH)

        correct = 0
        badly_predicted = []
        predicted_labels = []

        for i, test_instance in enumerate(self.x_val):
            test = np.array([test_instance])
            prob_dist = self.model.predict(test, verbose=0)[0]
            max_index = np.argmax(prob_dist)
            if self.y_val[i][max_index] == 1:
                # correctly classified instance
                correct += 1
            else:
                badly_predicted += [i]
                predicted_labels += [max_index]

        accuracy = correct * 1.0 / len(self.y_val)
        return badly_predicted, predicted_labels, accuracy
