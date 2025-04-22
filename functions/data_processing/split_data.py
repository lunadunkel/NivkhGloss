import pandas as pd
from collections import Counter
import random

def split_data(data):
    """
    Разделение данных по выборкам
    Пропорциональное сохранее баланса по текстам
    train: 75-85% данных
    eval, test: 10-15% данных каждый
    """

    df = pd.DataFrame(data)
    len_text = {d: [] for d in Counter(df['id'])}
    for m in df.itertuples():
        len_text[m.id].append(m)
    # len_text: группирование по id текстам

    random.seed(52)
    all_data = {'train': [], 'eval': [], 'test': []}
    for key, value in len_text.items():
        check = {'train': 0, 'eval': 0, 'test': 0}
        texts = [value[x].original for x in range(len(value))]
        random.shuffle(texts)
        number = len('\t'.join(texts).split('\t'))
        # print(f'Words in {key}: {number}')
        sample = 'train'
        for v in value:
            words = v.original.split()
            if sample == 'train' and 0.75 * number < check[sample] < 0.85 * number:
                sample = 'eval'
            elif sample == 'eval' and 0.1 * number < check[sample] < 0.15 * number:
                sample = 'test'
            check[sample] += len(words)
            formatted = {'original': v.original,
                'segmented': v.segmented,
                'glossed': v.glossed,
                'translation': v.translation,
                'id': v.id,
                'sent': v.sent,
                'bio_tag': v.bio_tag}
            all_data[sample].append(formatted)
    return all_data['train'], all_data['eval'], all_data['test']
