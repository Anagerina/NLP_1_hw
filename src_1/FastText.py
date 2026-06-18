import numpy as np
import torch
import torch.nn as nn


class MyEmbedTable:
    "Класс конструктора таблицы эмбеддингов, который будет использовать для FastText для создания эмбеддингов"

    def __init__(self, vocab: dict, fast_model):
        self.vocab = vocab
        self.fast_model = fast_model
        self.vocab_size = len(vocab)
        self.embed_dim = fast_model.vector_size

    def build_matrix(self):
        matrix = np.zeros((self.vocab_size, self.embed_dim))

        for token, idx in self.vocab.items():
            if token.lower() in ("<pad>"):
                continue

            if token not in self.fast_model:
                matrix[idx] = np.random.normal(
                    loc=0, scale=0.06, size=(self.embed_dim,)
                )

            else:
                matrix[idx] = self.fast_model[token]

        return matrix

    def make_emb_layer(self, padding_idx=0):
        matrix = self.build_matrix()
        tensor_matrix = torch.FloatTensor(matrix)

        embedding_layer = nn.Embedding.from_pretrained(
            tensor_matrix, freeze=True, padding_idx=padding_idx
        )

        return embedding_layer
