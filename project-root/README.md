# Customer Support Assistant

AI-based Customer Support Assistant using NLP, CNN, and Reinforcement Learning.

---

## Overview

This project develops an intelligent system that classifies and assists customer support queries using machine learning models such as baseline classifiers, Convolutional Neural Networks (CNN), and a Reinforcement Learning (RL) prototype.

---

## Dataset

The project uses the **Customer Support on Twitter** dataset.

Due to file size limits, the full dataset is not included in this repository.

* Kaggle Source:
  https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter

* Google Drive (Full Dataset):
  https://drive.google.com/drive/folders/1FjeuOT-CilEfIYtDJzlwBwc7bQC8A_Ex?usp=sharing

A sample dataset is included for testing:

```
data/sample_dataset.csv
```

---

## Quick Start

1. Install dependencies:

```
pip install -r requirements.txt
```

2. Run the project:

```
python train.py
```

---

## Project Structure

```
customer-support-assistant/
│── README.md
│── LICENSE
│── requirements.txt
│── run.sh
│
│── data/
│   └── sample_dataset.csv
│
│── src/
│   └── run_pipeline.py
│
│── models/
│   ├── baseline.py
│   ├── cnn_prototype.py
│   └── rl_stub.py
│
│── train.py
```

---

## Highlights

* NLP preprocessing pipeline
* Baseline machine learning models
* CNN-based classification
* Reinforcement Learning prototype
* Modular and extensible design

---

## Progress

* Data collection and preprocessing completed
* Exploratory Data Analysis (EDA) started
* Baseline models implemented
* CNN prototype running
* NLP pipeline established
* RL agent (basic version)

---

## Team

* Basilio, Leif Levinson C.
* Cirineo, Darrel Cedric R.
* Dingal, Marion Anthony S.
* Perez, Sebastian T.

---

## Notes

* Use the full dataset for better performance
* Sample dataset is for testing purposes only

---
