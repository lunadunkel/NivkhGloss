import torch
from torch.utils.data import Dataset

def prepare_sequence(seq, to_ix):
    return torch.tensor([to_ix[word] if word in to_ix else to_ix['<UNK>'] for word in seq], dtype=torch.long)

class PosDataset(Dataset):
    def __init__(self, sentences, word_dict, label_dict, char_dict, device='cpu'):
        self.sentences = sentences
        self.device = device
        self.word_dict = word_dict
        self.label_dict = label_dict
        self.char_dict = char_dict

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        answer = dict()
        answer['input_ids'] = prepare_sequence(sentence['input'].split('\t'), self.word_dict).to(self.device)
        answer['char_ids'] = prepare_sequence(list(sentence['input'].replace('\t', '#')), self.char_dict).to(self.device)
        answer['labels'] = prepare_sequence(sentence['label'].split('\t'), self.char_dict).to(self.device)
        answer['mask'] = answer['input_ids'] != 0
        answer['index']
        return answer