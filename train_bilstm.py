"""
Reference PyTorch BiLSTM training architecture.

Expected CSV columns:
text,label

Labels:
Bored, Confident, Confused, Curious, Frustrated

This script demonstrates the real deep-learning model architecture required by
the project. A sufficiently labeled dataset is required for meaningful training.
"""
import re
import json
import pandas as pd
import torch
import torch.nn as nn
from collections import Counter
from torch.utils.data import Dataset, DataLoader

LABELS = ["Bored", "Confident", "Confused", "Curious", "Frustrated"]
LABEL_TO_ID = {x: i for i, x in enumerate(LABELS)}

def tokenize(text):
    return re.findall(r"[a-z']+", str(text).lower())

class EmotionDataset(Dataset):
    def __init__(self, df, vocab, max_len=64):
        self.df, self.vocab, self.max_len = df, vocab, max_len
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ids = [self.vocab.get(t, 1) for t in tokenize(row["text"])][:self.max_len]
        ids += [0] * (self.max_len - len(ids))
        return torch.tensor(ids), torch.tensor(LABEL_TO_ID[row["label"]])

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, len(LABELS))
    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        features = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.fc(self.dropout(features))

def main():
    df = pd.read_csv("data/training_data.csv").dropna()
    counter = Counter(t for text in df["text"] for t in tokenize(text))
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for token, count in counter.items():
        if count >= 1:
            vocab[token] = len(vocab)

    loader = DataLoader(EmotionDataset(df, vocab), batch_size=16, shuffle=True)
    model = BiLSTMClassifier(len(vocab))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(20):
        model.train()
        total = 0
        for x, y in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"Epoch {epoch+1:02d} loss={total/len(loader):.4f}")

    torch.save(model.state_dict(), "models/bilstm_emotion.pt")
    with open("models/vocab.json", "w") as f:
        json.dump(vocab, f)
    print("Saved model and vocabulary.")

if __name__ == "__main__":
    main()
