import re

def make_bio(segm):
    bio_tag = []
    new_begining = True
    for letter in segm:
        if re.match('-', letter):
            new_begining = True
            continue

        if re.match('\t', letter):
            new_begining = True
            bio_tag.append('O')
            continue

        if re.match('[\t:\[\].,:;!?]', letter):
            bio_tag.append('O')
            continue

        if new_begining:
            bio_tag.append('B')
            new_begining = False
        else:
            bio_tag.append('I')
            
    segm_len = len(list(segm.replace('-', '')))
    bio_len = len(bio_tag)
    if segm_len != bio_len:
        raise Exception(f'Lengths of the segmentation and BIO-tag should be the same, got {segm_len} and {bio_len}')
    else:
        return bio_tag
