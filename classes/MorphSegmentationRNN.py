import torch
import torch.nn as nn
from NivkhGloss.classes.BasicNeuralClassifier import BasicNeuralClassifier

class MorphSegmentationRNN(BasicNeuralClassifier):

    def build_network(self, vocab_size, labels_number, n_layers=1, embed_dim=32, hidden_dim=128, num_heads=4,
                 dropout=0.0, use_crf=False, use_attention=False, aggregate_mode="last"):

        self.n_layers = n_layers # количество слоев
        self.hidden_dim = hidden_dim # размерность скрытого слоя
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0).to(self.device)
        self.aggregate_mode = aggregate_mode # функция аггрегации ПОКА НЕ РАБОТАЕТ
        self.use_crf = use_crf
        self.use_attention = use_attention

        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=n_layers,
                            bidirectional=True, batch_first=True, dropout=dropout).to(self.device) # слой лстм

        self.dropout = nn.Dropout(dropout) # дропаут по стандарту нулевой

        if self.use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=2 * hidden_dim,
                num_heads=num_heads,
                batch_first=True
            ).to(self.device) # слой внимания

        self.dense = nn.Linear(hidden_dim * 2, labels_number).to(self.device)

        if self.use_crf:
            self.crf = CRF(labels_number).to(self.device) # слой CRF

        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, inputs, mask=None):
        inputs = self.embedding(inputs)
        lstm_out, _ = self.lstm(inputs)
        lstm_out = self.dropout(lstm_out)
        if self.use_attention: # применение attention
            output, _ = self.attention(lstm_out, lstm_out, lstm_out)
        else:
            output = lstm_out
        logits = self.dense(output)

        if self.use_crf:
            return {"logits": logits}
        else:
            log_probs = self.log_softmax(logits)
            return {"log_probs": log_probs}
