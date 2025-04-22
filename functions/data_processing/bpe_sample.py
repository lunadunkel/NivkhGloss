def sample_with_bpe(sp, sample):
    new_sample = []
    for sent in sample:
        new_dict = {'original': sent['original'],
                    'segmented': sent['segmented'],
                    'glossed': sent['glossed'],
                    'translation': sent['translation'],
                    'metadata': sent['metadata'],
                    'id': sent['id'],
                    'sent': sent['sent'],
                    'bio_tag': sent['bio_tag']}
        bpe_subwords = sp.encode_as_pieces(sent['original'])
        sentence = (sent['original'])
        boundary_labels = get_bpe_boundary_labels(sentence, bpe_subwords)
        new_dict['bpe_tag'] = boundary_labels
        new_sample.append(new_dict)
    return new_sample
