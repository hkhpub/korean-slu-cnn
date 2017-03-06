"""
copyright Kwangho Heo
2016-12-05
"""
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from sklearn.cross_validation import train_test_split

import numpy as np
import argparse
from dialog import Utter, Dialog
from bs4 import BeautifulSoup, element
import os
from slu_sa_cnn import SLU_SA_CNN
import pickle

# np.random.seed(4331)  # for reproducibility           # acc: 95.6399%
# np.random.seed(6331)  # for reproducibility             # acc: 95.406%

MAX_SEQUENCE_LENGTH = 100
VALIDATION_SPLIT = 0.1


def main():
    parser = argparse.ArgumentParser(description='cnn classifier')
    parser.add_argument('--dataroot', dest='dataroot', action='store', required=True)
    parser.add_argument('--filenm', dest='filenm', action='store', required=True)
    args = parser.parse_args()

    sessions = []
    dialog = None
    utter = None

    print 'reading tagged file...'
    with open(os.path.join(args.dataroot, args.filenm)) as f:
        xmlstr = ' '.join(f.xreadlines())
        xmlstr = xmlstr.replace('\r\n', '')
        soup = BeautifulSoup(xmlstr, "lxml")

        for dialog_tag in soup.body.children:
            if type(dialog_tag) == element.NavigableString:
                continue

            dialog = Dialog()
            sessions += [dialog]

            for utter_tag in dialog_tag.children:
                if utter_tag.name == 'dial_id':
                    # print utter_tag.contents
                    pass
                elif utter_tag.name == 'dom':
                    pass
                elif utter_tag.name == 'utt':
                    utter = Utter()
                    utter.parse(utter_tag)
                    dialog.utters += [utter]

    print 'Generating n-grams...'
    for i in range(len(sessions)):
        dialog = sessions[i]
        for utter in dialog.utters:
            utter.generate_ngram(ngram=1)
    print 'Done'

    # create all texts (utterances)
    # create all labels (speech acts)
    maxlen = 0
    utterances = []
    # features
    texts = []
    prev_labels = []
    speakers = []
    utter_segments = []

    # labels
    utter_labels = []
    labels_index = {}
    for i in range(len(sessions)):
        dialog = sessions[i]
        for j, utter in enumerate(dialog.utters):
            label_id = len(labels_index)
            if utter.speech_act not in labels_index:
                labels_index[utter.speech_act] = label_id

            if maxlen < len(utter.ngram_tokens):
                maxlen = len(utter.ngram_tokens)

            features = ' '.join(utter.ngram_tokens).encode('utf-8')
            if j == 0:
                # first utterance's speech act
                prev_label = '<start>'
            else:
                prev_label = '<%d>' % utter_labels[-1:][0]

            prev_labels += [prev_label]
            speakers += [utter.speaker]
            utter_segments += [utter.segment]
            texts += [features]

            utterances += [utter]
            utter_labels += [labels_index[utter.speech_act]]

    labels_dict = dict((v, k) for k, v in labels_index.iteritems())

    print 'Found %s texts. Max sequence length: %d' % (len(texts), maxlen)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    for prev_label in set(prev_labels):
        word_index[prev_label] = len(word_index)
    print 'Found %s unique tokens.' % len(word_index)

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    labels = to_categorical(np.asarray(utter_labels))
    print 'Shape of data tensor:', data.shape
    print 'Shape of label tensor:', labels.shape

    # adding speech act label of previous turn
    # for i, feature_x in enumerate(data):
    #     feature_x[0] = word_index[prev_labels[i]]
    #     pass

    # split data to train set and validation set
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    # save indices
    with open('shuffled_indices.pickle', 'w') as f:
        pickle.dump(indices, f)
        f.close()

    data = data[indices]
    labels = labels[indices]
    nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

    x_train = data[:-nb_validation_samples]
    y_train = labels[:-nb_validation_samples]
    x_val = data[-nb_validation_samples:]
    y_val = labels[-nb_validation_samples:]

    for i in range(50):
        seed = int(np.random.random() * 10000)
        np.random.seed(seed)
        cnn_classifier = SLU_SA_CNN(word_index, x_train, y_train, x_val, y_val, sequence_length=MAX_SEQUENCE_LENGTH)
        cnn_classifier.train()
        badly_predicted, predicted_labels, accuracy = cnn_classifier.evaluate()
        with open('../results/accuracy.txt', 'a') as wf:
            wf.write('Random seed: %d\t\t Accuracy: %4f\n' % (seed, accuracy * 100.0))
            wf.close()


if __name__ == '__main__':
    main()
