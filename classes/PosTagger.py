import torch
import torch.nn as nn

class PosTagger(nn.Module):
    def __init__(self, word_embedding_dim, char_embedding_dim, hidden_dim, vocab_size, char_vocab_size, labels_number, use_char_ids=False, dropout=0.0,
                 aggregate_mode="last", device='cuda'):
        super(PosTagger, self).__init__()

        self.word_embedding = nn.Embedding(vocab_size, word_embedding_dim, padding_idx=0)

        self.char_embedding = nn.Embedding(char_vocab_size, char_embedding_dim, padding_idx=0)
        self.char_lstm = nn.LSTM(char_embedding_dim, char_embedding_dim, bidirectional=True, batch_first=True)
        total_embedding_dim = word_embedding_dim + 2 * char_embedding_dim if use_char_ids else word_embedding_dim
        self.lstm = nn.LSTM(total_embedding_dim, hidden_dim, bidirectional=True, batch_first=True)

        self.device = device
        self.dropout = nn.Dropout(dropout)

        self.dense = nn.Linear(hidden_dim * 2, labels_number).to(self.device)

        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, word_ids, char_ids=None, labels=None, mask=None):

        word_emb = self.word_embedding(word_ids)  # (batch, seq, word_emb_dim)

        if char_ids is not None:
            char_emb = self.char_embedding(char_ids)
            char_emb, _ = self.char_lstm(char_emb)

            embeddings = torch.cat([word_emb, char_emb], dim=-1)  # (batch, seq, total_emb_dim)
            embeddings = self.dropout(embeddings)
        else:
            embeddings = self.dropout(word_emb)

        output, _ = self.lstm(embeddings)  # (batch, seq, hidden_dim * 2)
        output = self.dropout(output)

        logits = self.dense(output)  # (batch, seq, labels_num)
        log_probs = self.log_softmax(logits)

        outputs = {"log_probs": log_probs}
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            logits_flat = logits.view(-1, logits.shape[-1])
            labels_flat = labels.view(-1)

            active_loss = labels_flat != 0

            active_logits = logits_flat[active_loss]
            active_labels = labels_flat[active_loss]

            loss = loss_fct(active_logits, active_labels)
            outputs["loss"] = loss
        return outputs