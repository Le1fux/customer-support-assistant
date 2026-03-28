import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def main():
    # 1️⃣ Load dataset
    df = pd.read_csv("data/twcs.csv")
    print("Dataset loaded successfully!")
    print(df.head())

    # 2️⃣ Quick overview
    print("\nDataset Info:")
    print(df.info())

    print("\nMissing Values per Column:")
    print(df.isnull().sum())

    print("\nDescriptive Stats:")
    print(df.describe())

    # 3️⃣ Cleaning
    df = df[df["inbound"] == True].copy()
    df = df[df["text"].str.strip() != ""]
    print(f"\nCleaned dataset shape: {df.shape}")

    # 4️⃣ Train / Val / Test split
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    print(f"\nTrain: {train_df.shape}, Validation: {val_df.shape}, Test: {test_df.shape}")

    # 5️⃣ Save splits
    train_df.to_csv("data/train.csv", index=False)
    val_df.to_csv("data/val.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)

    print("\nTrain/Val/Test CSVs saved!")

    # 6️⃣ Visualization
    df['text_length'] = df['text'].str.len()

    plt.figure(figsize=(10, 5))
    plt.hist(df['text_length'], bins=50)
    plt.title("Customer Message Length Distribution")
    plt.xlabel("Number of characters")
    plt.ylabel("Frequency")
    plt.show()


if __name__ == "__main__":
    main()