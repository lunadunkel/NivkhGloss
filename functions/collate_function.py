import numpy as np
import torch

def collate_fn(samples, dtype=torch.int64, keys=None):
    if keys is None:
        keys = ["input_ids", "labels", "mask"]
    device = samples[0]["input_ids"].device
    lengths = [elem["input_ids"].shape[0] for elem in samples]
    L = max(elem["input_ids"].shape[0] for elem in samples)

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
