import re

def get_bpe_boundary_labels(sentence, bpe_subwords):
    characters = list(sentence)
    labels = [0] * len(characters)
    index = 0
    for subword in bpe_subwords:
        if characters[index] == '\t':
            index += 1
            continue
        subword = re.sub('▁', '', subword)
        subword_len = len(subword)
        labels[index] = 1
        index += len(subword)
    return labels
