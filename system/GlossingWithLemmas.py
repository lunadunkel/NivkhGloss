import re
from collections import deque
import numpy as np

class GlossingWithLemmas:

    def __init__(self, stem_vocab, trie, navec, spacy_lemmatizer):

        self.stem_vocab = stem_vocab
        self.trie = trie
        self.navec = navec
        self.spacy_lemmatizer = spacy_lemmatizer
        self.punctuation = '!\"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~'
    
    def _cosine_similarity(self, a, b):
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return dot_product / (norm_a * norm_b)

    def _translation_embedding(self, translation):
        """ возвращает эмбеддинг предложения """
        lemmatized_translation = [word.lemma_ for word in self.spacy_lemmatizer(translation) 
                    if word.text not in self.punctuation]
        mean_emb = np.mean([self.navec[word] for word in lemmatized_translation if word in self.navec], axis=0)
        return mean_emb

    def _find_in_stem_vocab(self, word, gloss, translation):
        mean_emb = self._translation_embedding(translation)
        poss_definitions = [x for num, x in enumerate(self.stem_vocab[word]['ru']) 
                    if self.stem_vocab[word]['pos'][num] == gloss]

        if len(poss_definitions) == 1: 
            cosine = 0.5
            real_word = poss_definitions[0]
            if real_word in self.navec:
                cosine = self._cosine_similarity(self.navec[real_word], mean_emb)
            return '.'.join(poss_definitions[0].split()), cosine

        max_similarity, true_word = None, None

        for word in poss_definitions:
            real_word = ' '.join(word.split('.')) 
            if real_word in self.navec:
                cosine = self._cosine_similarity(self.navec[real_word], mean_emb)
                if max_similarity is None or cosine > max_similarity:
                    max_similarity = cosine
                    true_word = '.'.join(real_word.split())
        if true_word is not None:
            return true_word, max_similarity
        return gloss, 0

    def find_russian_lemmas(self, dictionary, translation):
        segmentation = dictionary['segmentation'].split('\t')
        glossing = dictionary['glossing'].split('\t')

        poss_pos_tags = ['VERB', 'NOUN', 'PRON', 'FUNC', 'ADV', 'DET', 'NUM', 'PROPN', 'ADJ']
        prev_morpheme = None

        final_glosses = []
        for word, gloss in zip(segmentation, glossing):
            word_parts, gloss_parts = word.split('-'), gloss.split('-')
            word_glosses = []
            for w, g in zip(word_parts, gloss_parts):
                w = w.replace('.,:!?', '')
                if g == 'PROPN':
                    word_glosses.append(w)
                    continue

                if g not in poss_pos_tags:
                    word_glosses.append(g)
                    prev_morpheme = g
                    continue

                word_glosses = []
                candidates = self._find_candidates(w, prev_word=prev_morpheme, max_changes=0)
                possible_stems = []
                for cand in candidates:
                    answer = self._find_in_stem_vocab(cand[0], g, translation)
                    if answer[0] == g == 'VERB':
                        answer = self._find_in_stem_vocab(cand[0], 'ADJ', translation)
                    possible_stems.append(answer)
                
                if not possible_stems:
                    true_gloss = 'UNK'
                else:
                    true_gloss = sorted(possible_stems, key=lambda x: -x[1])[0][0]
                word_glosses.append(true_gloss)
            final_glosses.append('-'.join(word_glosses))

        final_glossing = '\t'.join(final_glosses)

        return final_glossing

    def _find_candidates(self, word, prev_word=None, max_changes=0):
        """
          Поиск кандидатов в Trie на основе чередований
          принимает нивхское слово, возвращает русские леммы, отсортированные по количеству изменений
          word: нивхское слово для поиска
          prev_word: default None; предыдущее слово, вызывающие чередование (не считается как перемена)
          max_changes: default 0; максимально возможное количество изменений
        """
        rules_second_char = { 'зсч': 'зсч', 'ғкх': 'ғкх'}
        if prev_word is None:
            prev_word = '#'
        prev_word = re.sub('ь$', '', prev_word)

        hyps = deque([(self.trie.root, "", 0)])
        candidates = set()

        while hyps:
            node, new_match, changes = hyps.pop()
            if changes > max_changes:
                continue
            idx = len(new_match)

            if idx == len(word):
                if new_match in self.stem_vocab:
                    candidates.add((new_match, changes))
                continue

            for a, child in node.edges.items():
                if not new_match:  
                    self._handle_first_char(word, prev_word, a, child, hyps)
                elif a == word[idx]:  
                    hyps.appendleft((child, new_match + a, changes))
                elif idx == 1 and new_match[0] in 'иэ':  
                    if any(a in group and word[idx] in match for group, match in rules_second_char.items()):
                        hyps.appendleft((child, new_match + a, changes))
        final_candidates = sorted(list(candidates), key=lambda x: x[1])
        return final_candidates
                        
    def _handle_first_char(self, word, prev_word, a, child, hyps):
        """ добавляет в deque возможные чередования """
        rules_first_char = {
              'stop_or_vowel': {'вф': 'п', 'р': 'т', 'р̌': 'т',
                  'зс': 'тчс', 'ғгх': 'к', 'ӻӽ': 'ӄ'},
              'nasal': {'вбп': 'п', 'д': 'т', 'р̌': 'т',
                  'зс': 'тчс', 'ғгх': 'к'}}
        stops = ['п', 'т', 'ч', 'к', 'ӄ', 'б', 'д', 'г', 'ӷ', 'л']
        vowels = ['и', 'э', 'а', 'ы', 'о', 'у', 'я', 'е']
        nasal = ['н', 'л', 'ң', 'у']

        first_letter = word[0]
        last_prev = prev_word[-1]

        if last_prev in stops or last_prev in vowels or prev_word.endswith('н’'):
            rules = rules_first_char['stop_or_vowel']
        elif last_prev in nasal:
            rules = rules_first_char['nasal']
        else:
            if a == 'й' and first_letter != 'й':
                hyps.appendleft((child, 'й', 0))
            elif a == first_letter:
                hyps.appendleft((child, a, 0))
            else:
                hyps.appendleft((child, a, 1))
            return

        for pattern, target in rules.items():
            if first_letter in pattern:
                if a == target or a in target:
                    hyps.appendleft((child, target, 0))
        return