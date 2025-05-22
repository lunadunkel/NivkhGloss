import torch
import torch.nn as nn
from NivkhGloss.models.segmentation.BasicNeuralClassifier import BasicNeuralClassifier

class MorphSegmentationRNN(BasicNeuralClassifier):

    def build_network(self, vocab_size, labels_number, n_layers=1, embed_dim=32, hidden_dim=128, num_heads=4,
                      dropout=0.0, use_attention=False, bpe_vocab_size=None, aggregate_mode="last"):

        self.n_layers = n_layers  # количество слоев
        self.hidden_dim = hidden_dim  # размерность скрытого слоя

        # инициализация эмбеддингов символов
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0).to(self.device)

        # инициализация BPE-эмбеддингов, если используется BPE
        if self.use_bpe and bpe_vocab_size is not None:
            self.bpe_embedding = nn.Embedding(bpe_vocab_size, embed_dim, padding_idx=0).to(self.device)

        self.aggregate_mode = aggregate_mode 
        self.use_attention = use_attention

        # инициализация LSTM
        self.lstm = nn.LSTM(
            2 * embed_dim if self.use_bpe else embed_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout
        ).to(self.device)

        self.dropout = nn.Dropout(dropout)  # дропаут по стандарту нулевой

        # инициализация слоя внимания
        if self.use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=2 * hidden_dim,
                num_heads=num_heads,
                batch_first=True
            ).to(self.device)

        # полносвязный слой
        self.dense = nn.Linear(hidden_dim * 2, labels_number).to(self.device)

        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input_ids, bpe_boundary_labels=None, mask=None):
        # получение эмбеддингов символов
        input_ids = self.embedding(input_ids)

        # если BPE используется, получаем BPE-эмбеддинги
        if self.use_bpe:
            if bpe_boundary_labels is None:
                raise KeyError('use_bpe=True: BPE boundary labels are required')
            else:
                bpe_embeddings = self.bpe_embedding(bpe_boundary_labels)
            
            # объединение эмбеддингов символов и меток BPE
            combined_embeddings = torch.cat([input_ids, bpe_embeddings], dim=-1)  # B * L * (2 * d_emb)
        else:
            combined_embeddings = input_ids

        # применение LSTM
        lstm_out, _ = self.lstm(combined_embeddings)
        lstm_out = self.dropout(lstm_out)

        # применение механизма внимания
        if self.use_attention:
            output, _ = self.attention(lstm_out, lstm_out, lstm_out)
        else:
            output = lstm_out

        if self.aggregate_mode == "last":
            # последний элемент последовательности
            if mask is not None:
                lengths = mask.sum(dim=1)  # Длины последовательностей
                batch_indices = torch.arange(output.size(0), device=self.device)
                aggregated_output = output[batch_indices, lengths - 1]
            else:
                aggregated_output = output[:, -1, :]  # Последний элемент
        elif self.aggregate_mode == "mean":
            # усреднение всех элементов последовательности
            if mask is not None:
                mask = mask.unsqueeze(-1)  # расширение маски для совместимости с выходом
                output = output * mask
                lengths = mask.sum(dim=1)  # Длины последовательностей
                aggregated_output = output.sum(dim=1) / lengths.clamp(min=1)
            else:
                aggregated_output = output.mean(dim=1)
        elif self.aggregate_mode == "max":
            # максимальное значение по всей последовательности
            if mask is not None:
                mask = mask.unsqueeze(-1)  
                output = output.masked_fill(~mask.bool(), float('-inf')) 
            aggregated_output = output.max(dim=1)[0]
        else:
            raise ValueError(f"Unsupported aggregation mode: {self.aggregate_mode}")

        logits = self.dense(output)

        log_probs = self.log_softmax(logits)
        return {"log_probs": log_probs}
