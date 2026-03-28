import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder

DATA_DIR = "data"

# Load data
train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
val_df = pd.read_csv(f"{DATA_DIR}/val.csv")
print("Data loaded!")

# Add target column: whether a response exists
train_df['has_response'] = train_df['response_tweet_id'].notnull()
val_df['has_response'] = val_df['response_tweet_id'].notnull()

print("=== Baseline 1: Logistic Regression ===")
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vect = vectorizer.fit_transform(train_df['text'])
X_val_vect = vectorizer.transform(val_df['text'])

y_train = train_df['has_response']
y_val = val_df['has_response']

clf = LogisticRegression(max_iter=200)
clf.fit(X_train_vect, y_train)
y_pred = clf.predict(X_val_vect)
print(classification_report(y_val, y_pred))

print("=== Baseline 2: Simple Feedforward NN (subset) ===")
subset_size = 20000 
train_subset = train_df.sample(subset_size, random_state=42)
val_subset = val_df.sample(min(5000, len(val_df)), random_state=42)

y_train_enc = torch.tensor(LabelEncoder().fit_transform(train_subset['has_response']))
y_val_enc = torch.tensor(LabelEncoder().fit_transform(val_subset['has_response']))

vectorizer = CountVectorizer(max_features=5000)
X_train_bow = torch.tensor(vectorizer.fit_transform(train_subset['text']).toarray(), dtype=torch.float32)
X_val_bow = torch.tensor(vectorizer.transform(val_subset['text']).toarray(), dtype=torch.float32)

train_dataset = TensorDataset(X_train_bow, y_train_enc)
val_dataset = TensorDataset(X_val_bow, y_val_enc)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

class SimpleNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.fc(x)

model = SimpleNN(5000, 2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    val_preds = model(X_val_bow).argmax(dim=1)
    print(classification_report(y_val_enc, val_preds, target_names=['False','True']))