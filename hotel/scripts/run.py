"""
copyright Kwangho Heo
2016-12-05
"""
import argparse
from dialog import Utter, Dialog
from slu_sa_cnn import SLU_SA_CNN


def main():
    parser = argparse.ArgumentParser(description='cnn classifier')
    parser.add_argument('--tagged', dest='tagged', action='store', required=True)
    parser.add_argument('--morph_tagged', dest='morph_tagged', action='store', required=True)
    args = parser.parse_args()

    sessions = []
    dialog = None
    utter = None

    print 'reading tagged file...'
    with open(args.tagged, 'r') as f:
        for line in f.xreadlines():
            line = line.rstrip('\r\n')
            if len(line) == 0:
                if utter is not None and utter.speaker is not None:
                    dialog.utters += [utter]
                utter = Utter()
                pass

            elif ';' in line:
                if dialog is not None:
                    sessions += [dialog]
                    utter = None
                dialog = Dialog()

            else:
                utter.parse(line)
                pass
    print 'Done'

    print 'reading morph file...'
    all_morphs = []
    morph_by_dialog = None
    morph_by_utter = None

    with open(args.morph_tagged, 'r') as f:
        for line in f.xreadlines():
            line = line.rstrip('\r\n')

            if len(line) == 0:
                if morph_by_utter is not None and len(morph_by_utter) > 0:
                    morph_by_dialog += [morph_by_utter]
                morph_by_utter = []

            elif ';' in line:
                if morph_by_dialog is not None:
                    all_morphs += [morph_by_dialog]
                morph_by_dialog = []
                pass

            else:
                morphs = line.split('\t')[1:]
                morph_by_utter += morphs
                pass
    print 'Done'

    print 'Merging morphs...'
    for i in range(len(sessions)):
        dialog = sessions[i]
        morph_by_dialog = all_morphs[i]
        for j, utter in enumerate(dialog.utters):
            morph_by_utter = morph_by_dialog[j]
            utter.morphs = morph_by_utter
            pass
        pass
    print 'Done'

    print 'Generating n-grams...'
    for i in range(len(sessions)):
        dialog = sessions[i]
        for utter in dialog.utters:
            utter.generate_ngram(ngram=3)
    print 'Done'

    # print 'Dump ngrams...'
    # with open('unigram.txt', 'w') as wf:
    #     for i in range(len(sessions)):
    #         dialog = sessions[i]
    #         for utter in dialog.utters:
    #             wf.writelines(' '.join(utter.unigram_tokens))
    #             wf.writelines('\n')
    #
    # with open('bigram.txt', 'w') as wf:
    #     for i in range(len(sessions)):
    #         dialog = sessions[i]
    #         for utter in dialog.utters:
    #             wf.writelines(' '.join(utter.bigram_tokens))
    #             wf.writelines('\n')
    #
    # with open('trigram.txt', 'w') as wf:
    #     for i in range(len(sessions)):
    #         dialog = sessions[i]
    #         for utter in dialog.utters:
    #             wf.writelines(' '.join(utter.trigram_tokens))
    #             wf.writelines('\n')
    # print 'Done'

    # create all texts (utterances)
    # create all labels (speech acts)
    maxlen = 0
    texts = []
    prev_labels = []
    labels = []
    labels_index = {}
    for i in range(len(sessions)):
        dialog = sessions[i]
        for j, utter in enumerate(dialog.utters):
            label_id = len(labels_index)
            if utter.speech_act not in labels_index:
                labels_index[utter.speech_act] = label_id

            if maxlen < len(utter.ngram_tokens):
                maxlen = len(utter.ngram_tokens)

            features = ' '.join(utter.ngram_tokens)
            # first utterance's speech act
            if j == 0:
                prev_label = '<start>'
            else:
                prev_label = '<%d>' % labels[-1:][0]

            prev_labels += [prev_label]
            texts += [features]
            labels += [labels_index[utter.speech_act]]

    print 'Found %s texts. Max sequence length: %d' % (len(texts), maxlen)

    cnn_classifier = SLU_SA_CNN(texts, labels, labels_index, prev_labels)
    cnn_classifier.train()


if __name__ == '__main__':
    main()
