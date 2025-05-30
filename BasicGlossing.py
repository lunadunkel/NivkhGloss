import torch
import re

class BasicGlossing:

    def __init__(self, segm_model, pos_model, gloss_model,
                 symb_vocab, char_vocab, word_vocab,
                 pos_label_vocab, morpheme_vocab, gloss_vocab,
                 glosses_dictionary, device):
      """
        Система глоссирования принимает модели и словари для обработки слова и предсказания глоссов.

        Args:
            segm_model: модель сегментации слова на морфемы
            pos_model: модель предсказания частей речи
            gloss_model: модель предсказания глоссов
            symb_vocab: словарь символов для сегментации
            char_vocab: словарь символов для POS-модели
            word_vocab: словарь слов для POS-модели
            pos_label_vocab: словарь меток частей речи
            morpheme_vocab: словарь морфем
            gloss_vocab: словарь глосс
            glosses_dictionary: словарь с аффиксами для глоссирования
            device: устройство для вычислений ('cpu' или 'cuda')
        """

      # сегментация
      self.segm_model = segm_model
      self.symb_vocab = symb_vocab

      # POS-tagging
      self.pos_model = pos_model
      self.word_vocab = word_vocab
      self.char_vocab = char_vocab
      self.pos_label_vocab = {value: key for key, value in pos_label_vocab.items()}

      # rule-glossing & nn-glossing
      self.glosses_dictionary = glosses_dictionary
      self.gloss_model = gloss_model
      self.gloss_vocab = gloss_vocab
      self.morpheme_vocab = morpheme_vocab
      self.id2gloss = {key: value for value, key in self.gloss_vocab.items()}

      self.device = device

    def gloss_sent_no_lemmas(self, sent):
        """
            Глоссирование на уровне предложения

            sentence –> segmentation
            sent –> pos_tags
            segmentation + pos_tags –> rule_glosses
            rule_glosses –> lstm_glosses

        """

        segmentation = self._segment(sent)
        pos_tags = self._define_pos(sent)
        rule_glosses = self._rule_gloss_sent(segmentation, pos_tags)
        glossed = self._glossing(segmentation, rule_glosses)
        answer = {'segmentation': segmentation, 'glossing': glossed}
        return answer

    def _encode_sample(self, sequence, vocabulary):
            """ кодирование последовательностей на основе словаря """
            return torch.tensor([vocabulary[seq] if seq in vocabulary
                              else vocabulary['<UNK>'] for seq in sequence],
                                dtype=torch.int64).to(self.device)

    def _segment(self, sent):
        """ функция сегментации предложения """
        self.segm_model.eval()
        x = self._encode_sample(sent, self.symb_vocab)

        mask = (x != 0).bool().to(self.device)
        input_ids = x.unsqueeze(0)
        mask = mask.unsqueeze(0)

        with torch.no_grad():
            predictions = self.segm_model(input_ids, mask)['log_probs']

        labels = torch.argmax(predictions, dim=-1)
        labels = labels.view(-1)

        final_answer, prev_label = '', 0
        for label, char in zip(labels, sent):
            label = label.item()
            match label:
                case 0:
                    final_answer += char

                case 1:
                    if prev_label in [1, 2]:
                        final_answer += '-'
                    final_answer += char

                case 2:
                    final_answer += char
            prev_label = label
        return final_answer

    def _define_pos(self, sent):
        """ функция определения части речи пословно """
        self.pos_model.eval()
        pos_tags = []
        for word in sent.split('\t'):
            word_ids = self._encode_sample(word, self.word_vocab)
            char_ids = self._encode_sample(word, self.char_vocab)
            with torch.no_grad():
                output = self.pos_model(word_ids, char_ids)['log_probs']
            pos = torch.argmax(output, dim=-1).cpu().numpy()[0]
            pos_tags.append(self.pos_label_vocab[pos])
        return '\t'.join(pos_tags)

    def _find_correct_gloss(self, row, pos_tag, stem=True):
        """ поиск подходящих глосс для правилового PosTagger """
        true_gloss = []
        category, gl, pos = row['category'], row['glosses'], row['pos']
        for c, g, p in zip(category, gl, pos):
            if (not stem and c == 'prefix') and (p == pos_tag or p in ['INDEP', 'ANY']):
                true_gloss.append(g)
            elif (stem and c == 'suffix') and (p == pos_tag or p in ['INDEP', 'ANY']):
                true_gloss.append(g)
        return true_gloss

    def _rule_gloss_sent(self, segmentation, pos_tags):
        """ правиловый глоссатор """
        hierarchy = ['VERB', 'NOUN', 'PRON', 'PROPN', 'ADV', 'ADP', 'NUM', 'DET']
        glosses, true_pos_tags = [], []
        for word, pos_tag in zip(segmentation.split('\t'), pos_tags.split('\t')):
            morphemes = word.split('-')
            stem, word_glosses = False, []
            pos = pos_tag
            for morph in morphemes:
                if morph in self.glosses_dictionary:
                    row = self.glosses_dictionary[morph]
                    true_gloss = self._find_correct_gloss(row, pos, stem=stem)
                    if not true_gloss and not stem:
                        stem, true_gloss = True, ['<STEM>']
                    elif not true_gloss and stem:
                        for h in hierarchy:
                            true_gloss = self._find_correct_gloss(row, h)
                            if true_gloss:
                                pos = h
                                break
                        if not true_gloss: true_gloss = ['<UNK>']
                else:
                    stem, true_gloss = True, ['<STEM>']

                word_glosses.append('#'.join(true_gloss))

            true_pos_tags.append(pos)
            glosses.append('-'.join(word_glosses))

        final_gloss = []
        for gl, pos in zip(glosses, true_pos_tags):
            final_gloss.append(re.sub('<STEM>', pos, gl))
        return '\t'.join(final_gloss)

    def _glossing(self, segmentation, rule_glosses):
        self.gloss_model.eval()
        segmentation = segmentation.replace('\t', '\t<TAB>\t').replace('-', '\t').split('\t')
        candidates = [x.split('#') for x in rule_glosses.replace('\t', '\t<TAB>\t').replace('-', '\t').split('\t')]
        target_index = torch.tensor([num for num, x in enumerate(candidates) if len(x) > 1]).to('cuda')

        if target_index.nelement() == 0:
            return '-'.join(x[0] for x in candidates).replace('-<TAB>-', '\t')

        cand_mask = torch.zeros(len(target_index), len(self.gloss_vocab), dtype=torch.bool).to('cuda')
        input_ids = self._encode_sample(segmentation, self.morpheme_vocab).unsqueeze(0)
        for j, cand_idx in enumerate(target_index):
            positions = self._encode_sample(candidates[cand_idx], self.gloss_vocab)
            for idx in positions:
                if 0 <= idx < len(self.gloss_vocab):
                    cand_mask[j, idx] = True

        with torch.no_grad():
            logits = self.gloss_model(input_ids)[0][target_index]
            masked_logits = logits.masked_fill(~cand_mask, -1e9)
            for position, ten in zip(target_index, masked_logits):
                pred = torch.argmax(ten).item()
                candidates[position] = [self.id2gloss[pred]]
        return '-'.join(x[0] for x in candidates).replace('-<TAB>-', '\t')