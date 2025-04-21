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
            if 'bpe_boundary_labels' in bpe_boundary_labels:
                bpe_boundary_labels = batch['bpe_boundary_labels'].to(device)
                outputs = model(input_ids, bpe_boundary_labels=bpe_boundary_labels, mask=mask)
            else:
                outputs = model(input_ids, mask=mask)
            if model.use_crf:
                crf_loss = -model.crf(outputs['logits'], labels, mask).mean()
                ce_loss = criterion(outputs['logits'].view(-1, outputs['logits'].size(-1)), labels.view(-1))
                loss = crf_loss + ce_loss
            else:
                loss = model.criterion(
                    outputs['log_probs'].view(-1, outputs['log_probs'].size(-1)),
                    labels.view(-1)
                )
                loss = (loss * mask.view(-1)).sum() / mask.sum()
            total_loss += loss.item()

            if model.use_crf:
                preds = model.crf.viterbi_decode(outputs['logits'], mask)
            else:
                preds = torch.argmax(outputs['log_probs'], dim=-1).cpu().tolist()

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())
            all_masks.extend(mask.cpu().tolist())
    accuracy = model.calculate_accuracy(all_labels, all_preds, all_masks)
    word_level_accuracy = model.calculate_word_level_accuracy(all_labels, all_preds)
    avg_loss = total_loss / len(data_loader)
    return avg_loss, accuracy, word_level_accuracy
