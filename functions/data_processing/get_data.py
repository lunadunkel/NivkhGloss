import os
import re

def get_data(text_name, path="''"):
    data = []
    text = path + '/' + text_name
    pos = 1
    with open(text, 'r', encoding='utf8') as file:
        new_dict = {'original': '', 'segmented': '', 'glossed': '', 'translation': None, 'metadata': None, 'id': text_name, 'sent': pos}
        for string in file:
            if re.findall('\d+(\_\d+)*=', string):
                pos += 1
                new_dict['translation'] = re.search('(?<=\d=[\t ]).*', string).group()
                data.append(new_dict)
                new_dict = {'original': '', 'segmented': '', 'glossed': '', 'translation': None, 'metadata': None, 'id': text_name, 'sent': pos}
                continue
            if re.findall('\d+\>', string):
                substring = re.search('(?<=\>[ \t]).*', string).group()
                substring = re.sub('\t+', '\t', substring)
                new_dict['segmented'] += '\t'.join([substring])
                new_dict['original'] += '\t'.join([substring]).replace('-', '')
                continue
            if re.findall('\d+\<', string):
                substring = re.search('(?<=\<[ \t]).*', string).group()
                substring = re.sub('\t+', '\t', substring)
                new_dict['glossed'] += '\t'.join([substring])
            if re.findall('#', string):
                string = re.sub('(?<=#) *', '', string)
                if new_dict['metadata'] is None:
                    new_dict['metadata'] = '\n'.join([re.search('(?<=#).*', string).group()])
                else:
                    new_dict['metadata'] += '\n' + '\n'.join([re.search('(?<=#).*', string).group()])
    return data
