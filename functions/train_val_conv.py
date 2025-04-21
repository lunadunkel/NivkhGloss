import os
import torch
from tqdm import tqdm

def validate_model(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_masks = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            mask = batch.get('mask', None)

            outputs = model(input_ids, mask=mask)
            log_probs = outputs["log_probs"]

            log_probs_flat = log_probs.view(-1, log_probs.size(-1))  # (B * L, C)
            labels_flat = labels.view(-1)  # (B * L)

            if mask is not None:
                mask_flat = mask.view(-1)  # (B * L)

            labels_flat = torch.where(mask_flat.bool(), labels_flat, torch.tensor(-100).to(labels.device))


            loss = model.criterion(log_probs_flat, labels_flat)
            total_loss += loss.item()

            preds = torch.argmax(outputs['log_probs'], dim=-1).cpu().tolist()

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())
            all_masks.extend(mask.cpu().tolist())

    accuracy = model.calculate_accuracy(all_labels, all_preds, all_masks)
    word_level_accuracy = model.calculate_word_level_accuracy(all_labels, all_preds)
    avg_loss = total_loss / len(data_loader)
    return avg_loss, accuracy, word_level_accuracy


def train_model(model, optimizer, train_loader, val_loader,
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
            mask = batch.get('mask', None)
            outputs = model(input_ids, mask=mask)
            log_probs = outputs["log_probs"]

            log_probs_flat = log_probs.view(-1, log_probs.size(-1))  # (B * L, C)
            labels_flat = labels.view(-1)  # (B * L)

            if mask is not None:
                mask_flat = mask.view(-1)  # (B * L)

            labels_flat = torch.where(mask_flat.bool(), labels_flat, torch.tensor(-100).to(labels.device))


            loss = model.criterion(log_probs_flat, labels_flat)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        val_loss, val_accuracy, word_level_accuracy = validate_model(model, val_loader, device)
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
