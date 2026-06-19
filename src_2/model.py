import torch
import torch.nn as nn
import torch.nn.functional as F

PAD_IDX = 0


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output):

        attn_weights = self.attn(lstm_output)
        attn_weights = F.softmax(attn_weights, dim=1)
        context = torch.sum(lstm_output * attn_weights, dim=1)
        return context


class CNNBiLSTM(nn.Module):
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

        # BiLSTM
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True
        )
        self.attention = Attention(hidden_dim * 2)

        # TextCNN
        self.conv3 = nn.Conv1d(
            in_channels=embed_dim, out_channels=128, kernel_size=3, padding=1
        )
        self.conv4 = nn.Conv1d(
            in_channels=embed_dim, out_channels=128, kernel_size=4, padding=2
        )
        self.conv5 = nn.Conv1d(
            in_channels=embed_dim, out_channels=128, kernel_size=5, padding=2
        )

        combined_dim = (hidden_dim * 2) + (128 * 3)

        self.bn = nn.BatchNorm1d(num_features=combined_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(combined_dim, num_classes)

    def forward(self, x, mask):
        embeds = self.embedding(x)

        lstm_out, _ = self.lstm(embeds)
        attn_scores = self.attention.attn(lstm_out)

        mask_expanded = mask.unsqueeze(-1)
        attn_scores = attn_scores.masked_fill(~mask_expanded, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=1)
        lstm_features = torch.sum(lstm_out * attn_weights, dim=1)

        cnn_in = embeds.permute(0, 2, 1)

        c3 = F.relu(self.conv3(cnn_in))
        p3 = F.adaptive_max_pool1d(c3, 1).squeeze(2)

        c4 = F.relu(self.conv4(cnn_in))
        p4 = F.adaptive_max_pool1d(c4, 1).squeeze(2)

        c5 = F.relu(self.conv5(cnn_in))
        p5 = F.adaptive_max_pool1d(c5, 1).squeeze(2)

        cnn_features = torch.cat([p3, p4, p5], dim=1)
        pooled = torch.cat([lstm_features, cnn_features], dim=1)
        normalized = self.bn(pooled)
        x = self.dropout(normalized)
        x = self.fc(x)
        return x


"""
class CNNBiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=300, hidden_dim=128, num_layers=1, num_classes=4, dropout=0.3, embedding_matrix=None, freeze_embeddings=False):
        super().__init__()
        self.embedding = embedding_matrix
        embed_dim = embedding_matrix.embedding_dim
        
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_dim*2)

        self.conv3 = nn.Conv1d(in_channels=embed_dim, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(in_channels=embed_dim, out_channels=128, kernel_size=4, padding=2)
        self.conv5 = nn.Conv1d(in_channels=embed_dim, out_channels=128, kernel_size=5, padding=2)

        combined_dim = (hidden_dim * 2) + (128 * 3)

        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(num_features=hidden_dim * 2)
        self.fc = nn.Linear(combined_dim * 2, num_classes)
    
    def forward(self, x, mask):

        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        attn_scores = self.attention.attn(lstm_out)
        
        mask_expanded = mask.unsqueeze(-1)
        attn_scores = attn_scores.masked_fill(~mask_expanded, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=1)
        context = torch.sum(lstm_out * attn_weights, dim=1)
        normalized = self.bn(context)
        x = self.dropout(normalized)
        x = self.fc(x)
        return x

        """ """
        mask_expanded = mask.unsqueeze(-1).expand_as(lstm_out)
        lstm_out_masked = lstm_out.masked_fill(~mask_expanded, float('-inf'))
        max_pooled, _ = torch.max(lstm_out_masked, dim=1)
        
        lstm_out_zeros = lstm_out.masked_fill(~mask_expanded, 0.0)
        actual_lengths = mask.sum(dim=1, keepdim=True)
        mean_pooled = lstm_out_zeros.sum(dim=1) / actual_lengths

        pooled = torch.cat([max_pooled, mean_pooled], dim=1)
        
        normalized = self.bn(pooled)

        x = self.dropout(normalized)
        x = self.fc(x)
        return x """
