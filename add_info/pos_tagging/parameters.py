from NivkhGloss.add_info.pos_tagging import *

def open_json_dict(path):
    with open(path, 'r', encoding='utf8') as file:
        return json.load(file)

WORD_TO_IDX = open_json_dict('word_vocabulary.json')
CHR_TO_IDX = open_json_dict('char_vocabulary.json')
TAG_TO_IDX = open_json_dict('label_vocabulary.json')

VOCAB_SIZE = len(WORD_TO_IDX)
WORD_EMBEDDING_DIM = 64
CHAR_EMBEDDING_DIM = 32
CHAR_VOCAB_SIZE = len(CHR_TO_IDX)
HIDDEN_DIM = 128
OUTPUT_DIM = len(TAG_TO_IDX)
BATCH_SIZE = 16
EPOCHS = 10