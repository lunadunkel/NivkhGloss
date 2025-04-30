import torch

def validate_model(model, data_loader, device, use_bpe=False):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_masks = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            mask = model._prepare_mask(input_ids, batch.get('mask'))
            if use_bpe:
                if 'bpe_boundary_labels' in batch:
                    bpe_boundary_labels = batch['bpe_boundary_labels'].to(device)
                    outputs = model(input_ids, bpe_boundary_labels=bpe_boundary_labels, mask=mask)
                else:
                    raise KeyError("BPE labels required in dataset")
            else:
                outputs = model(input_ids, mask=mask)

            loss = model.criterion(
                outputs['log_probs'].view(-1, outputs['log_probs'].size(-1)),
                labels.view(-1)
            )
            loss = (loss * mask.view(-1)).sum() / mask.sum()
            total_loss += loss.item()

            preds = torch.argmax(outputs['log_probs'], dim=-1).cpu().tolist()
            labels = labels.cpu().tolist()
            mask = mask.cpu().tolist()

            for p, l, m in zip(preds, labels, mask):
                if len(p) != len(l) or len(p) != len(m):
                    raise ValueError(f"Length mismatch: preds={len(p)}, labels={len(l)}, mask={len(m)}")

            all_preds.extend(preds)
            all_labels.extend(labels)
            all_masks.extend(mask)
    accuracy = model.calculate_accuracy(all_labels, all_preds, all_masks)
    word_level_accuracy = model.calculate_word_level_accuracy(all_labels, all_preds)
    avg_loss = total_loss / len(data_loader)
    return avg_loss, accuracy, word_level_accuracy
