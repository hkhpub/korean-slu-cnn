"""
copyright Kwangho Heo
2016-12-05
"""
import argparse
from dialog import Utter, Dialog
from bs4 import BeautifulSoup, element
import os
from slu_sa_cnn import SLU_SA_CNN


def main():
    parser = argparse.ArgumentParser(description='cnn classifier')
    parser.add_argument('--dataroot', dest='dataroot', action='store', required=True)
    parser.add_argument('--filenm', dest='filenm', action='store', required=True)
    args = parser.parse_args()

    sessions = []
    dialog = None
    utter = None

    print 'reading tagged file...'
    for i in range(5):
        filenm = '%s%d.txt' % (args.filenm, i+1)
        print 'processing %s' % filenm
        with open(os.path.join(args.dataroot, filenm)) as f:
            xmlstr = ' '.join(f.xreadlines())
            xmlstr = xmlstr.replace('\r\n', '')
            soup = BeautifulSoup(xmlstr)

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

            features = ' '.join(utter.ngram_tokens).encode('utf-8')
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
