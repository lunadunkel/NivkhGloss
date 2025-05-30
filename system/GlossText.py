from NivkhGloss.system.GlossingWithLemmas import *
from NivkhGloss.system.BasicGlossing import *

class GlossText:

    def __init__(self, segm_model, pos_model, gloss_model,
                 symb_vocab, char_vocab, word_vocab, pos_label_vocab,
                 morpheme_vocab, gloss_vocab, glosses_dictionary,
                 stem_vocabulary, trie, navec, spacy_lemmatizer, device='cuda'):

        self.Glosser = BasicGlossing(segm_model, pos_model, gloss_model, symb_vocab, char_vocab, word_vocab,
                 pos_label_vocab, morpheme_vocab, gloss_vocab, glosses_dictionary, device)

        self.LemmaGlosser = GlossingWithLemmas(stem_vocabulary, trie, navec, spacy_lemmatizer)


    def gloss_text(self, text, translation=None):
        """
        глоссирование на уровне текста:
        text: список предложений
        translation: список переводов
        """
        final_glossing = []
        if translation is None:
            translation = [None * len(text)] 
        for num, (sent, trans) in enumerate(zip(text, translation)):
            answer = self.gloss_sent(sent, trans)
            segmentation = f'{num}>\t' + answer['segmentation']
            glossing = f'{num}<\t' + answer['glossing']
            final_glossing.append('\n'.join([segmentation, glossing]))
        return final_glossing

    def gloss_sent(self, sent, translation=None):
        """
        глоссирование на уровне предложения:
        sent: предложение
        translation: перевод
        """
        dictionary = self.Glosser.gloss_sent_no_lemmas(sent)
        glossed = dictionary['glossing']
        if translation is not None:
            glossed = self.LemmaGlosser.find_russian_lemmas(dictionary, translation)
        answer = {'segmentation': dictionary['segmentation'],
                  'glossing': glossed}
        return answer