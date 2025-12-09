import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score
from torch.nn.utils.rnn import pad_sequence
from collections import Counter

class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab=None, max_vocab_size=30000):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)

        if vocab is None:
            self.vocab = self.build_vocab(texts, max_vocab_size)
        else:
            self.vocab = vocab

    def build_vocab(self, texts, max_vocab_size):
        counter = Counter()
        for t in texts:
            counter.update(t.lower().split())

        vocab = {"<PAD>": 0, "<UNK>": 1}
        for word, _ in counter.most_common(max_vocab_size - 2):
            vocab[word] = len(vocab)
        return vocab

    def encode(self, text):
        return torch.tensor(
            [self.vocab.get(w, self.vocab["<UNK>"]) for w in text.lower().split()],
            dtype=torch.long
        )

    def __getitem__(self, idx):
        return self.encode(self.texts[idx]), self.labels[idx]

    def __len__(self):
        return len(self.labels)


def collate_fn(batch):
    texts, labels = zip(*batch)
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)
    labels = torch.tensor(labels)
    return texts_padded, labels


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_classes=2, kernel_sizes=(3,4,5), num_filters=100):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim,
                      out_channels=num_filters,
                      kernel_size=k)
            for k in kernel_sizes
        ])

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)

        conv_outputs = []
        for conv in self.convs:
            c = torch.relu(conv(x))
            p = torch.max(c, dim=2)[0]
            conv_outputs.append(p)

        x = torch.cat(conv_outputs, dim=1)
        x = self.dropout(x)
        return self.fc(x)


def compute_metrics(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            preds = model(X).argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())

    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro")
    }


def train_model(model, train_loader, val_loader, device, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.to(device)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Train Loss: {total_loss/len(train_loader):.4f}")

        metrics = compute_metrics(model, val_loader, device)
        print(f" Accuracy:       {metrics['accuracy']:.4f}")
        print(f" Balanced Acc:   {metrics['balanced_acc']:.4f}")
        print(f" F1 (macro):     {metrics['f1_macro']:.4f}")


if __name__ == "__main__":
    data = pd.read_csv("HateSpeechDataset.csv")
    # data["Label"] = data["Label"].astype(int)
    data = data[data["Label"].isin(['0', '1'])]

    texts = data["Content"].astype(str).tolist()
    labels = data["Label"].astype(int).tolist()

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, stratify=labels
    )

    train_dataset = TextDataset(train_texts, train_labels)
    val_dataset = TextDataset(val_texts, val_labels, vocab=train_dataset.vocab)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TextCNN(vocab_size=len(train_dataset.vocab))

    print("Training on:", device)
    train_model(model, train_loader, val_loader, device, epochs=5)
