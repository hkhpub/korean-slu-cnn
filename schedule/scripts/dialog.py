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
    raw_morph = None
    morphs = None

    def __init__(self):
        self.ngram_tokens = []
        self.unigram_tokens = []
        self.bigram_tokens = []
        self.trigram_tokens = []
        pass

    def parse(self, utter_bs4_tag):
        # fill in parsing logic
        for element in utter_bs4_tag.children:
            if element.name == 'spk':
                self.speaker = element.contents[0]
            elif element.name == 'snt':
                self.text = element.contents[0]
            elif element.name == 'mor':
                self.raw_morph = element.contents[0]
                tokens = self.raw_morph.split()
                morph_tokens = [token for token in tokens if '/' in token]
                morphs = []
                for morph_token in morph_tokens:
                    morphs += morph_token.split('+')
                self.morphs = morphs

            elif element.name == 'sa':
                self.speech_act = element.contents[0]
                self.speech_act = self.speech_act.replace(' ', '')
            elif element.name == 'ds':
                self.segment = element.contents[0]
            else:
                pass
        pass

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
