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

# скачивание моделей
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

with open('NivkhGloss/models/lemma_insert/trie.pkl', 'rb') as f:
    trie = pickle.load(f)

with open('NivkhGloss/stem_vocab.json') as file:
    stem_vocab = json.load(file)

NAVEC_URL = "https://storage.yandexcloud.net/natasha-navec/packs/navec_hudlit_v1_12B_500K_300d_100q.tar"
SPACY_MODEL = "ru_core_news_sm"

def install_dependencies():
    """Устанавливает конкретные зависимости: navec и spacy"""
    packages = [
        "navec",
        "spacy"
    ]

    for package in packages:
        print(f"Установка {package}...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", package],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"Ошибка при установке {package}\n{result.stdout}")
        else:
            print(f"{package} установлен")

def download_spacy_model():
    import spacy
    print(f"Установка модели {SPACY_MODEL}...")
    subprocess.check_call([sys.executable, "-m", "spacy", "download", SPACY_MODEL])
    print(f"{SPACY_MODEL} установлен")
    nlp = spacy.load(SPACY_MODEL)
    return nlp

def download_navec():
    path = "navec_hudlit_v1_12B_500K_300d_100q.tar"
    response = requests.get(NAVEC_URL, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(path, "wb") as f:
        for data in tqdm(response.iter_content(chunk_size=1024),
                             desc="Загрузка Navec модели",
                             total=total_size // 1024 + 1,
                             unit='KB',
                             ncols=80):
            if data:
                f.write(data)
    from navec import Navec
    navec = Navec.load(path)
    return navec


install_dependencies()
nlp = download_spacy_model()
navec = download_navec()