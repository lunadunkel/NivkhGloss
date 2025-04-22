import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import accuracy_score
from TorchCRF import CRF

class BasicNeuralClassifier(nn.Module):
    """ Базовый классификатор
    включает возможность:
    - CRF-слой
    - BPE-слой """
    
    def __init__(self, vocab_size, labels_number, bpe_vocab_size=None, use_bpe=False, device="cpu", criterion=nn.NLLLoss(reduction="mean"), **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.labels_number = labels_number
        self.use_bpe = use_bpe  # использование BPE
        if self.use_bpe and bpe_vocab_size is not None:
            self.bpe_vocab_size = bpe_vocab_size
        self.device = device
        
        # учет BPE
        if self.use_bpe and bpe_vocab_size is not None:
            self.build_network(vocab_size, labels_number, bpe_vocab_size=bpe_vocab_size, **kwargs)
        else:
            self.build_network(vocab_size, labels_number, **kwargs)
        
        self.criterion = criterion

    def build_network(self, vocab_size, labels_number, bpe_vocab_size=None, **kwargs):
        raise NotImplementedError("You should implement network construction in your derived class.")

    def forward(self, input_ids, bpe_boundary_labels=None, mask=None):
        raise NotImplementedError("You should implement forward pass in your derived class.")

    def predict(self, input_ids, bpe_boundary_labels=None, mask=None):
        self.eval()
        with torch.no_grad():
            input_ids = input_ids.to(self.device)
            mask = self._prepare_mask(input_ids, mask)
            
            # если BPE используется, передаем bpe_boundary_labels
            if self.use_bpe:
                if bpe_boundary_labels is not None:
                    bpe_boundary_labels = bpe_boundary_labels.to(self.device)
                    outputs = self(input_ids, bpe_boundary_labels=bpe_boundary_labels, mask=mask)
                else:
                    raise KeyError('use_bpe=True: BPE boundary labels are required')
            else:
                outputs = self(input_ids, mask=mask)

            if self.use_crf:  # использование CRF
                return self.crf.viterbi_decode(outputs["logits"], mask)
            else:
                preds = torch.argmax(outputs["log_probs"], dim=-1).cpu().tolist()
                return [pred[:torch.sum(m).item()] for pred, m in zip(preds, mask)]

    def _prepare_mask(self, input_ids, mask):
        if mask is None:
            mask = (input_ids != 0)  # Маска: True для ненулевых элементов
        return mask.to(self.device).bool()

    def calculate_accuracy(self, labels, predictions, mask):
        true_labels = []
        true_preds = []
        for preds, lbls, m in zip(predictions, labels, mask):
            if m is None:
                active = [True] * len(lbls)
            elif isinstance(m, torch.Tensor):
                active = m.bool().tolist()  # тензор в булевый формат
            elif isinstance(m, list):
                active = [bool(x) for x in m]  # список в булевый формат
            else:
                raise ValueError(f"Unsupported mask type: {type(m)}")

            # проверка размерности
            if len(lbls) != len(active):
                raise ValueError("Lengths of labels and mask must match.")

            # истинные метки (с учетом маски)
            true_labels.extend([label for label, flag in zip(lbls, active) if flag])

            # CRF предсказания уже обрезаны
            if len(preds) < len(lbls):
                true_preds.extend(preds)
            else:
                # фильтр предсказаний по маске
                if len(preds) != len(active):
                    raise ValueError("Lengths of predictions and mask must match.")
                true_preds.extend([pred for pred, flag in zip(preds, active) if flag])

        return accuracy_score(true_labels, true_preds)

    def calculate_word_level_accuracy(self, true_labels, predictions):
        correct_words = 0
        total_words = 0
        for preds, lbls in zip(predictions, true_labels):
            word_boundaries = [-1]
            mask_start = True
            for num, p in enumerate(lbls):
                if mask_start is False and p == 0:
                    word_boundaries.append(num)
                    mask_start = True
                elif p != 0:
                    mask_start = False
            for i in range(len(word_boundaries)-1):
                a, b = word_boundaries[i]+1, word_boundaries[i+1]
                word = lbls[a:b]
                pred = preds[a:b]
                if word == pred:
                    correct_words += 1
                total_words += 1

        return correct_words / total_words if total_words > 0 else 0.0
