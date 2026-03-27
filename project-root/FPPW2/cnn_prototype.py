import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, f1_score
import numpy as np


train_df = pd.read_csv("train.csv").sample(50000, random_state=42)  
val_df   = pd.read_csv("val.csv").sample(10000, random_state=42)
print("Subset loaded for CNN prototype!")

le = LabelEncoder()
y_train = torch.tensor(le.fit_transform(train_df['inbound']))
y_val   = torch.tensor(le.transform(val_df['inbound']))


ablations = [
    {"max_features": 500,  "kernel_size": 3},
    {"max_features": 1000, "kernel_size": 3},
    {"max_features": 500,  "kernel_size": 5},
]

results_table = []


class SimpleCNN(nn.Module):
    def __init__(self, input_dim, num_classes, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=kernel_size, padding=kernel_size//2)
        self.relu  = nn.ReLU()
        self.pool  = nn.MaxPool1d(2)
        self.fc    = nn.Linear((input_dim//2)*16, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1) 
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


for ab in ablations:
    print(f"\nRunning ablation: max_features={ab['max_features']}, kernel_size={ab['kernel_size']}")
    
    vectorizer = CountVectorizer(max_features=ab['max_features'])
    X_train = torch.tensor(vectorizer.fit_transform(train_df['text']).toarray(), dtype=torch.float32)
    X_val   = torch.tensor(vectorizer.transform(val_df['text']).toarray(), dtype=torch.float32)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset   = TensorDataset(X_val, y_val)
    train_loader  = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader    = DataLoader(val_dataset, batch_size=64)

    model = SimpleCNN(input_dim=ab['max_features'], num_classes=len(le.classes_), kernel_size=ab['kernel_size'])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
    print(f"Epoch done, last batch loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        val_preds = model(X_val).argmax(dim=1)

    f1 = f1_score(y_val, val_preds, average='macro')
    print(f"Validation F1: {f1:.4f}")
    results_table.append({"max_features": ab['max_features'], "kernel_size": ab['kernel_size'], "val_f1": f1})

 
val_text = val_df['text'].values
lengths = np.array([len(t.split()) for t in val_text])
short_mask  = lengths < 10
medium_mask = (lengths >= 10) & (lengths < 20)
long_mask   = lengths >= 20

for name, mask in [("short", short_mask), ("medium", medium_mask), ("long", long_mask)]:
    if mask.sum() == 0:
        print(f"Skipping {name} tweets: no samples in this slice")
        continue
    slice_f1 = f1_score(y_val[mask], val_preds[mask], average='macro', zero_division=0)
    print(f"F1 on {name} tweets ({mask.sum()} samples): {slice_f1:.4f}")

for inbound_flag in [True, False]:
    mask = (val_df['inbound'].values == inbound_flag)
    if mask.sum() == 0:
        print(f"Skipping {'inbound' if inbound_flag else 'outbound'} tweets: no samples")
        continue
    slice_f1 = f1_score(y_val[mask], val_preds[mask], average='macro', zero_division=0)
    print(f"F1 on {'inbound' if inbound_flag else 'outbound'} tweets ({mask.sum()} samples): {slice_f1:.4f}")


print("\nAblation Results Summary:")
for r in results_table:
    print(r)