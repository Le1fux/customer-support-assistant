import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer

train_df = pd.read_csv("train.csv").sample(5000, random_state=42)
val_df   = pd.read_csv("val.csv").sample(1000, random_state=42)
print("Subset loaded for CNN prototype!")

le = LabelEncoder()
y_train = torch.tensor(le.fit_transform(train_df['inbound']))
y_val   = torch.tensor(le.transform(val_df['inbound']))

vectorizer = CountVectorizer(max_features=500)  
X_train = torch.tensor(vectorizer.fit_transform(train_df['text']).toarray(), dtype=torch.float32)
X_val   = torch.tensor(vectorizer.transform(val_df['text']).toarray(), dtype=torch.float32)

train_dataset = TensorDataset(X_train, y_train)
val_dataset   = TensorDataset(X_val, y_val)
train_loader  = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader    = DataLoader(val_dataset, batch_size=32)

class SimpleCNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
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

model = SimpleCNN(500, len(le.classes_))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} done, last batch loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    val_preds = model(X_val).argmax(dim=1)

from sklearn.metrics import classification_report

class_names = [str(c) for c in le.classes_]

print("\nCNN Prototype Results:")
print(classification_report(y_val, val_preds, target_names=class_names))
