import re

def create_glossing_dataset(data):
    final_data = []
    for sent in data:
        segm = re.sub('\ufeff', '', sent['segmented'])
        segm = re.sub('^\t', '', segm)
        segm = re.sub('[\,\?\!\;«\']+', '\t', segm)
        gloss = re.sub('[\,\?\!\;«\']+', '\t', sent['glossed'])
        gloss = re.sub('^\t', '', gloss)

        gold_segm = re.sub('\t', '\t<SEP>\t', segm)
        gold_gloss = re.sub('\t', '\t<SEP>\t', gloss)

        gold_segm = re.sub('-', '\t<HYPH>\t', gold_segm)
        gold_gloss = re.sub('-', '\t<HYPH>\t', gold_gloss)

        gold_gloss = re.sub('(?:[А-яё\.]+[А-яё])(?!\.CAUSEE|\.DU|\.EXCL|\.INCL)', '<STEM>', gold_gloss)
        gold_gloss = re.sub('(?:<STEM>\.)+<STEM>', '<STEM>', gold_gloss)
        gold_gloss = re.sub('(?<=[A-z])\.(?=(?:$| |\t))', '', gold_gloss)
        dictionary = {'segmented': gold_segm.split('\t'),
                      'STEM_glossed': gold_gloss.split('\t'),
                      'id': sent['id'],
                      'sents': sent['sents']}
        final_data.append(dictionary)
    return final_data
