import torch
import torch.nn as nn

class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim,
                 num_layers=1, bidirectional=False, dropout=0.0, device='cpu'):
        super(BiLSTMTagger, self).__init__()

        self.device = device
        self.embedding = nn.Embedding(vocab_size, embed_dim).to(self.device)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        ).to(self.device)

        self.hidden_dim = hidden_dim * 2 if bidirectional else hidden_dim

        self.dense = nn.Linear(self.hidden_dim,
                               output_dim).to(self.device)
        self.dropout = nn.Dropout()

    def forward(self, input_ids, mask=None):
        if mask is None:
            mask = (input_ids != 0)

        lengths = mask.sum(dim=1).cpu()

        embeddings = self.embedding(input_ids)
        embeddings = self.dropout(embeddings)

        packed_embeddings = nn.utils.rnn.pack_padded_sequence(
            embeddings, lengths, batch_first=True, enforce_sorted=False
        )
        lengths = mask.sum(dim=1).cpu()
        packed_lstm_out, _ = self.lstm(packed_embeddings)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_lstm_out, batch_first=True)

        logits = self.dense(lstm_out)
        return logits