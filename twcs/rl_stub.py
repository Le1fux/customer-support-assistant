import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report


train_df = pd.read_csv("train.csv")
val_df = pd.read_csv("val.csv")
print("Train/Val loaded for RL stub!")


le = LabelEncoder()
y_train = le.fit_transform(train_df['inbound']) 
y_val = le.transform(val_df['inbound'])


num_batches = 100
rewards = []

for batch in range(num_batches):
    reward = np.random.rand() * 0.1 + batch * 0.01
    rewards.append(reward)

print("RL stub training done!")


plt.figure(figsize=(8,5))
plt.plot(range(1, num_batches+1), rewards, marker='o')
plt.title("RL Stub Reward Curve")
plt.xlabel("Batch")
plt.ylabel("Reward")
plt.grid(True)
plt.tight_layout()
plt.show()


val_preds = np.ones_like(y_val)

labels = np.unique(y_val)
target_names = [str(c) for c in labels]

print("\nRL Stub Results (placeholder classification):")
print(classification_report(y_val, val_preds, labels=labels, target_names=target_names))
