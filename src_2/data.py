import re
from collections import Counter

import torch
from datasets import load_dataset
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from config import CLASS_NAMES


def get_data():

    print("Loading AG News dataset...")
    dataset = load_dataset("fancyzhx/ag_news", cache_dir="../ag_news_cache")

    print(f"\nDataset structure:")
    print(dataset)
    print(f"\nTrain set size: {len(dataset['train'])}")
    print(f"Test set size: {len(dataset['test'])}")

    train_texts = dataset["train"]["text"]
    train_labels = dataset["train"]["label"]
    test_texts = dataset["test"]["text"]
    test_labels = dataset["test"]["label"]

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts,
        train_labels,
        test_size=0.05,
        random_state=42,
        stratify=train_labels,
    )

    print(f"\nAfter split:")
    print(f"Train: {len(train_texts)}")
    print(f"Val: {len(val_texts)}")
    print(f"Test: {len(test_texts)}")

    class_names = CLASS_NAMES
    print(f"\nClasses: {class_names}")

    return (
        dataset,
        train_texts,
        val_texts,
        train_labels,
        val_labels,
        test_texts,
        test_labels,
        class_names,
    )


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_IDX = 0
UNK_IDX = 1


def clean_text(text):
    text = text.replace("\\", " ")
    text = text.replace("39;s", "'s")
    text = re.sub(r"&[a-z]+;", " ", text)
    return text


def simple_tokenize(text, stop=None):
    text = clean_text(text)
    text = text.lower()

    tokens = re.findall(r"\b[a-z0-9]+(?:-[a-z0-9]+)*(?:\'[a-z]+)?\b", text)
    if stop is not None:
        tokens = [token for token in tokens if token not in stop]
    return tokens


def make_token_dict(train_texts, vocab_size=2, max_vocab_size=50000):
    print("Building vocabulary...")
    word_counter = Counter()
    for text in train_texts:
        tokens = simple_tokenize(text)
        word_counter.update(tokens)

    # top_30_pairs = word_counter.most_common(30)
    # top_30_stop_words = [word for word, count in top_30_pairs]

    # for word in MY_STOP_WORDS:
    # if word in word_counter:
    # del word_counter[word]

    print(f"Total unique words: {len(word_counter)}")
    print(
        f"Words with freq >= 2: {sum(1 for count in word_counter.values() if count >= 2)}"
    )

    vocab = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}

    vocab_size = vocab_size

    max_vocab_size = max_vocab_size
    for word, count in word_counter.most_common():
        if count >= 2 and vocab_size < max_vocab_size:
            vocab[word] = vocab_size
            vocab_size += 1

    print(f"\nVocabulary size: {len(vocab)}")
    print(f"Top 10 words: {list(vocab.keys())[:10]}")

    idx_to_word = {idx: word for word, idx in vocab.items()}

    return (
        vocab,
        idx_to_word,
    )  # top_30_pairs


class AGNewsDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=128, stop_words=None):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
        self.stop_words = stop_words

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        tokens = simple_tokenize(text, self.stop_words)
        indices = [self.vocab.get(token, UNK_IDX) for token in tokens]

        if len(indices) > self.max_len:
            indices = indices[: self.max_len]

        return torch.tensor(indices, dtype=torch.long), torch.tensor(
            label, dtype=torch.long
        )


def collate_fn(batch):
    texts, labels = zip(*batch)
    lengths = [len(text) for text in texts]
    max_len = max(lengths)

    padded_texts = []
    masks = []
    for text in texts:
        pad_length = max_len - len(text)
        padded = torch.cat([text, torch.full((pad_length,), PAD_IDX, dtype=torch.long)])
        mask = torch.cat(
            [
                torch.ones(len(text), dtype=torch.bool),
                torch.zeros(pad_length, dtype=torch.bool),
            ]
        )
        padded_texts.append(padded)
        masks.append(mask)

    return torch.stack(padded_texts), torch.stack(masks), torch.stack(labels)
