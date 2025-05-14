import os
import torch
from tqdm import tqdm
from NivkhGloss.models.segmentation.train_eval.validation import validate_model

def train_model(model, optimizer, train_loader, val_loader, use_bpe=False, 
    device="cpu", checkpoint_dir="checkpoints", num_epochs=25, patience=3):

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_accuracy = 0.0
    early_stopping_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()
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
            mask = mask.view(-1)
            loss = (loss * mask).sum() / mask.sum()

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # валидация
        val_loss, val_accuracy, word_level_accuracy = validate_model(model, val_loader, device, use_bpe=use_bpe)
        print(f"\nValidation - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, Word-level: {word_level_accuracy:.4f}")

        if word_level_accuracy > best_accuracy:
            best_accuracy = word_level_accuracy
            checkpoint_path = os.path.join(checkpoint_dir, f"best_model_epoch{epoch+1}_acc{best_accuracy:.4f}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'accuracy': best_accuracy
            }, checkpoint_path)
            print(f"New best model saved to: {checkpoint_path}")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print("Early stopping triggered")
                print(f"Best accuracy: {best_accuracy}")
                break
