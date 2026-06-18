import torch
import torch.nn as nn

PAD_IDX = 0


class BiLSTM(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=300,
        hidden_dim=128,
        num_layers=1,
        num_classes=4,
        dropout=0.3,
        embedding_matrix=None,
        freeze_embeddings=False,
    ):
        super().__init__()
        self.embedding = embedding_matrix
        embed_dim = embedding_matrix.embedding_dim

        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(num_features=hidden_dim * 4)
        self.fc = nn.Linear(hidden_dim * 4, num_classes)

    def forward(self, x, mask):
        x = self.embedding(x)

        lstm_out, _ = self.lstm(x)

        mask_expanded = mask.unsqueeze(-1).expand_as(lstm_out)
        lstm_out_masked = lstm_out.masked_fill(~mask_expanded, float("-inf"))
        max_pooled, _ = torch.max(lstm_out_masked, dim=1)

        lstm_out_zeros = lstm_out.masked_fill(~mask_expanded, 0.0)
        actual_lengths = mask.sum(dim=1, keepdim=True)
        mean_pooled = lstm_out_zeros.sum(dim=1) / actual_lengths

        pooled = torch.cat([max_pooled, mean_pooled], dim=1)

        normalized = self.bn(pooled)

        x = self.dropout(normalized)
        x = self.fc(x)
        return x
