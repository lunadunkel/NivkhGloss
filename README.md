# Система глоссирования для нивхского языка

Система включает в себя следующие этапы:
- Сегментация через дефис;
- Определение POS-тегов;
- Правиловое глоссирование на основе грамматических данных;
- Нейросетевой классификатор: выбор между предсказанными правиловой системой глоссами.

> [!TIP] 
> Для инференса можно использовать HF Spaces
> По [ссылке](https://huggingface.co/spaces/lunadunkel/NivkhGloss) можно опробовать модель на CPU.

Для более удобного пользования (в том числе глоссирования целого текста) лучше использовать colab. Для запуска необходимо:

1. Клонировать репозиторий
```
git clone https://github.com/lunadunkel/NivkhGloss.git
```

2. Установить необходимые пакеты...

```
from NivkhGloss.system.installation import *
from NivkhGloss.system.models import *
```

...и объявить необходимые переменные – NLP (spaCy для лемматизации) и navec (русские эмбеддинги) 
```
nlp = download_spacy_model()
navec = download_navec()
```

4. Также необходимо объявить бор (trie) и словарь основ (stem_vocab)
```
with open('NivkhGloss/models/lemma_insert/trie.pkl', 'rb') as f:
    trie = pickle.load(f)

with open('NivkhGloss/vocabularies/glossing/stem_vocab.json') as file:
    stem_vocab = json.load(file)
```

5. Для объявления каждой модели используйте следующий код:
```
segm_dir = 'bpe_attention_lstm_cnn'

segm_model = MorphSegmentationCNN(vocab_size=len(symb_vocab),
    labels_number=len(label_dict), hidden_dim=512, n_layers=3, dropout=0.4,
    device="cuda", window=(3, 6), bpe_vocab_size=2500, use_attention=True,
    use_lstm=True, use_bpe=True).to("cuda")

model_state_dict = torch.load('/content/NivkhGloss/models/pth/bpe_attention_lstm_cnn.pth')['model_state_dict']
segm_model.load_state_dict(model_state_dict)

postagging_dir = 'char_pos_tagging'

pos_model = PosTagger(word_embedding_dim=64, char_embedding_dim=32,
                         hidden_dim=128,  vocab_size=len(word_vocab),
                         char_vocab_size=len(char_vocab), labels_number=len(pos_label_vocab),
                         device="cuda", use_char_ids=True, dropout=0.2).to(device="cuda")


model_state_dict = torch.load('/content/NivkhGloss/models/pth/char_pos_tagging.pth')['model_state_dict']
pos_model.load_state_dict(model_state_dict)

glossing_dir = 'glossing'

gloss_model = BiLSTMTagger(len(morpheme_vocab), embed_dim=128,
                     dropout=0.5, bidirectional=False,
                     num_layers=2, hidden_dim=256,
                     output_dim=len(gloss_vocab), device="cuda").to("cuda")


model_state_dict = torch.load('/content/NivkhGloss/models/pth/glossing.pth')['model_state_dict']
gloss_model.load_state_dict(model_state_dict)
```

6. Импорт модели и ее объявление

```
from NivkhGloss.system.GlossText import *
NivkhGlosser = GlossText(segm_model, pos_model, gloss_model,
                         symb_vocab, char_vocab, word_vocab,
                         pos_label_vocab, morpheme_vocab, gloss_vocab,
                         glosses_dictionary, stem_vocab,
                         trie, navec, nlp, device='cuda')
```

7. Для лучшего результата также рекомендуется предобратывать текст

```
def clear_punctuation(line):
    line = '\t'.join(line.split())
    new_line = re.sub('\[[А-ЯЁ]+\:\]\t', '', line)
    new_line = re.sub('[\.\,\?\!\:\(\)]+', '', new_line)
    new_line = re.sub('^\t|\t$', '', new_line)
    new_line = new_line.lower()
    return new_line
```

## Использование глоссатора

После предыдущих шагов можно использовать глоссатор. Глоссатор включает в себя две функции:
- глоссирование по предложению;
- глоссирование всего текста.

Пример:
> NivkhGlosser.gloss_sent('ӿы,	ӿарор̌	мулкхир̌	пыньх	лытр̌',
>                        translation='Да. Затем в берестяной корзинке бульон рыбный варит')
