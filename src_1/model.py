import torch
import torch.nn as nn
import torch.nn.functional as F

PAD_IDX = 0


class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=300,
        num_filters=100,
        filter_sizes=[3, 4, 5],
        num_classes=4,
        dropout=0.5,
        embedding_matrix=None,
        freeze_embeddings=False,
    ):

        super().__init__()
        self.embedding = embedding_matrix
        self.embed_dim = embedding_matrix.embedding_dim
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(self.embed_dim, num_filters, kernel_size=fs)
                for fs in filter_sizes
            ]
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        self.bn = nn.BatchNorm1d(num_features=len(filter_sizes) * num_filters)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).expand_as(x)
            x = x.masked_fill(~mask_expanded, 0.0)

        x = x.transpose(1, 2)  # B x d x T
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(x))
            pooled = F.max_pool1d(conv_out, kernel_size=conv_out.size(2)).squeeze(2)
            conv_outputs.append(pooled)

        x = torch.cat(conv_outputs, dim=1)
        x = self.bn(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x
