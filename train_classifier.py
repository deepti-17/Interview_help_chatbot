from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch
import json
import numpy as np

import os
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["CURL_CA_BUNDLE"] = ""

from transformers import BertTokenizer, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", trust_remote_code=True)

# Load intents
with open("intents.json", "r") as f:
    intents = json.load(f)

# Prepare data
texts = []
labels = []
label2id = {}
id2label = {}
for idx, intent in enumerate(intents["intents"]):
    label2id[intent["tag"]] = idx
    id2label[idx] = intent["tag"]
    for pattern in intent["patterns"]:
        texts.append(pattern)
        labels.append(idx)

# Tokenizer
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Custom Dataset
class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=64)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Split data
X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, random_state=42)
train_dataset = IntentDataset(X_train, y_train, tokenizer)
val_dataset = IntentDataset(X_val, y_val, tokenizer)

# Load model
# model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(label2id))
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(label2id) 
)

# Training
training_args = TrainingArguments(
    output_dir="./model",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=500,
    eval_steps=500,
    # evaluation_strategy="steps",
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()

# Save model
model.save_pretrained("model")
tokenizer.save_pretrained("model")

# Save label encoder
import pickle
with open("label_encoder.pkl", "wb") as f:
    pickle.dump((label2id, id2label), f)
