import torch
from NivkhGloss.add_info.char_dict import char_dict
from NivkhGloss.add_info.label_dict import label_dict

def encode_sample(word, labels, device='cpu'):
    x = torch.tensor([char_dict[char] for char in word], dtype=torch.int64).to(device)
    y = torch.tensor([label_dict[label] for label in labels], dtype=torch.int64).to(device)
    return x, y

def make_dataset(data, use_bpe=False):
    dataset = []
    for idx, sent in enumerate(data):
        original = sent['segmented'].replace('-', '')
        x, y = encode_sample(original, sent['bio_tag'], device='cuda')
        mask = (x != 0)
        dictionary = {'index': idx,'input_ids': x, 'labels': y, 'mask': mask}
        if use_bpe: 
            dictionary['bpe_boundary_labels'] = sent['bpe_tag']
        dataset.append(dictionary)
    return dataset
