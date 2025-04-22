import re
import os

def clear_data(path, filename):
    new_data = []
    full_path = f'{path}/{filename}'
    with open(full_path, 'r', encoding='utf8') as file:
        segmented, glossed, numbers = [], [], []

        for line in file:
            line = line.strip()
            if line == '' or re.search('#', line):
                continue
            
            if re.search('\d+\>', line):
                numbers.append(re.search('\d+', line).group())
                line = re.sub('\d+\>', '', line)
                line = re.sub('\[[А-ЯЁ]+\:\]', '', line)
                line = re.sub('\(.*\)', '', line)
                line = re.sub('[\.\,\?\!]+', '', line)
                word_in_line = line.split()
                segmented.append('\t'.join(word_in_line))

            if re.search('\d+\<', line):
                line = re.sub('\d+\<', '', line)
                line = re.sub('\[[А-ЯЁ]+\:\]', '', line)
                line = re.sub('\(.*\)', '', line)
                word_in_line = line.split()
                glossed.append('\t'.join(word_in_line))

            if re.search('\d+\=', line):
                segmentation = '\t'.join(segmented)
                if segmentation == '':
                    segmented, glossed, numbers = [], [], []
                    continue
                glossing = '\t'.join(glossed)
                line = re.sub('_+', '', line)
                line = re.sub('\d+\=\t+', '', line)
                dictionary = {'original': segmentation.replace('-', ''),
                              'segmented': segmentation,
                              'glossed': glossing,
                              'translation': line,
                              'id': filename,
                              'sents':  numbers}
                segmented, glossed, numbers = [], [], []

                new_data.append(dictionary)
                
    return new_data
