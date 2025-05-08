from sklearn.metrics import accuracy_score
from tqdm import tqdm
import numpy as np
import torch

def compute_accuracy(preds, labels, ignore_index=0):
    preds = preds.view(-1).cpu().numpy()
    labels = labels.view(-1).cpu().numpy()
    mask = labels != ignore_index
    filtered_preds = preds[mask]
    filtered_labels = labels[mask]

    return accuracy_score(filtered_labels, filtered_preds)

def validate(model, val_dataloader, device='cpu', use_char_emb=False):
    model.eval()
    total_loss = 0
    total_acc = 0
    count = 0
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to('cuda')
            labels = batch['labels'].to('cuda')
            if use_char_emb:
                char_ids = batch['char_ids']
                output = model(input_ids, char_ids=char_ids, labels=labels)
            else:
                output = model(input_ids, labels=labels)
            preds = torch.argmax(output['log_probs'], dim=-1)
            loss = output['loss']
            acc = compute_accuracy(preds, labels, ignore_index=0)
            total_acc += acc * input_ids.size(0)
            count += input_ids.size(0)
            total_loss += loss.item()

        avg_loss = total_loss / len(val_dataloader)
        avg_acc = total_acc / count

    return avg_loss, avg_acc

def train_model(model, optimizer, train_loader, val_loader, use_char_emb=False, 
    device="cpu", checkpoint_dir="checkpoints", num_epochs=10, patience=3):

    os.makedirs(checkpoint_dir, exist_ok=True)
    total_loss = 0
    total_acc = 0
    count = 0
    best_accuracy = 0.0
    early_stopping_counter = 0

    for epoch in range(num_epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            mask = batch['mask'].to(device)

            optimizer.zero_grad()

            if use_char_emb:
                if 'char_ids' in batch:
                    char_ids = batch['char_ids'].to(device)
                    outputs = model(input_ids, char_ids=char_ids, labels=labels, mask=mask)
                else: 
                    raise KeyError("CHAR-ids required in dataset")
            else:
                outputs = model(input_ids, mask=mask, labels=labels,)

            loss = outputs['loss']
            loss.backward()
            optimizer.step()
              
            preds = torch.argmax(outputs['log_probs'], dim=-1)
            acc = compute_accuracy(preds, labels, ignore_index=0)
            total_acc += acc * input_ids.size(0)
            count += input_ids.size(0)
            total_loss += loss.item()

        avg_acc = total_acc / count
        avg_loss = total_loss / len(train_dataloader)
        print(f"\nTraining - Loss: {avg_loss:.4f}, Average accuracy: {avg_acc:.4f}, Accuracy: {acc:.4f}")
        print('---------------------------------------------------------------------------------------')

        # валидация
        val_loss, val_accuracy = validate(model, val_loader, device, use_char_emb=use_char_emb)
        print(f"\nValidation - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
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