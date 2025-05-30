import os
import subprocess
import sys
import requests
from tqdm import tqdm
import re
import json
import requests
import random
import torch
import spacy
import numpy as np
import pickle
import torch.nn as nn
import pandas as pd
from glob import glob
from string import punctuation
from collections import deque
from huggingface_hub import hf_hub_download
from sklearn.metrics import accuracy_score
from google.colab import userdata
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from importlib import reload
from sklearn.metrics import classification_report

from NivkhGloss.models.segmentation.MorphSegmentationCNN import MorphSegmentationCNN
from NivkhGloss.models.pos_tagger.PosTagger import PosTagger
from NivkhGloss.models.glossing.GlossingLSTM import BiLSTMTagger
from NivkhGloss.models.lemma_insert.trie import *

# сегментация
from NivkhGloss.vocabularies.segmentation.symb_vocab import symb_vocab
from NivkhGloss.vocabularies.segmentation.label_dict import label_dict
# частеречный анализатор
from NivkhGloss.vocabularies.pos_tagging.word_vocab import word_vocab
from NivkhGloss.vocabularies.pos_tagging.pos_label_vocab import pos_label_vocab
from NivkhGloss.vocabularies.pos_tagging.char_vocab import char_vocab
# глоссирование
from NivkhGloss.vocabularies.glossing.gloss_vocab import gloss_vocab
from NivkhGloss.vocabularies.glossing.morpheme_vocab import morpheme_vocab
from NivkhGloss.vocabularies.glossing.glosses_dictionary import glosses_dictionary