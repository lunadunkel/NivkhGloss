import torch
import numpy as np

def collate_fn_pos(samples, use_char_ids=False, dtype=torch.int64):
    keys = ["input_ids", "labels", "mask"]
    max_by = 'input_ids'
    if use_char_ids:
        keys.append('char_ids')
        max_by = 'char_ids'

    device = samples[0]["input_ids"].device

    lengths = [elem[max_by].shape[0] for elem in samples]
    L = max(elem[max_by].shape[0] for elem in samples)

    answer = dict()
    for key in keys:
        answer[key] = torch.stack([
            torch.cat([
                elem[key],
                torch.zeros(size=(L-len(elem[key]),), dtype=dtype).to(device)
            ]) for elem in samples
        ])

    answer["index"] = np.array([elem["index"] for elem in samples])
    return answer