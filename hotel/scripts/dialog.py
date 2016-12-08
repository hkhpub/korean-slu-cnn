"""
copyright Kwangho Heo
2016-12-05
"""
from nltk import ngrams


class Utter:

    speaker = None
    text = None
    speech_act = None
    semantic_tag = None
    segment = None
    morphs = None

    def __init__(self):
        self.ngram_tokens = []
        self.unigram_tokens = []
        self.bigram_tokens = []
        self.trigram_tokens = []
        pass

    def parse(self, line):

        if '/SP/' in line:
            self.speaker = line.split('/SP/')[1]

        elif '/KS/' in line:
            self.text = line.split('/KS/')[1]

        elif '/SA/' in line:
            self.speech_act = line.split('/SA/')[1].rstrip(' *')

        elif '/ST/' in line:
            self.semantic_tag = line.split('/ST/')[1]

        elif '/DS/' in line:
            self.segment = line.split('/DS/')[1]

        else:
            print 'exception: %s' % line

    def generate_ngram(self, ngram):
        # unigram
        for morph in self.morphs:
            self.ngram_tokens += [morph]
            self.unigram_tokens += [morph]

        if ngram >= 2:
            # bigram
            bigrams = ngrams(self.morphs, 2, pad_left=True, pad_right=True, left_pad_symbol='<S>', right_pad_symbol='</S>')
            for grams in bigrams:
                bigram = '%s__%s' % grams
                self.ngram_tokens += [bigram]
                self.bigram_tokens += [bigram]

        if ngram >= 3:
            trigrams = ngrams(self.morphs, 3, pad_left=True, pad_right=True, left_pad_symbol='<S>', right_pad_symbol='</S>')
            for grams in trigrams:
                trigram = '%s__%s__%s' % grams
                self.ngram_tokens += [trigram]
                self.trigram_tokens += [trigram]


class Dialog:

    def __init__(self):
        self.utters = []
        pass
