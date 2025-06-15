import pandas as pd
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from sklearn.preprocessing import LabelEncoder
import os
os.makedirs("model", exist_ok=True) 

# Load your dataset
df = pd.read_csv("intents_data.csv")  # Make sure this file has 'text' and 'label' columns

# Encode the labels
le = LabelEncoder()
df["label"] = le.fit_transform(df["label"])  # Saves labels as 0-7
label2id = {label: i for i, label in enumerate(le.classes_)}
id2label = {i: label for label, i in label2id.items()}

# Save for later use in chatbot.py
import json
with open("model/label_mappings.json", "w") as f:
    json.dump({"label2id": label2id, "id2label": id2label}, f)

# Tokenize
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
def tokenize(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)

dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.2)
dataset = dataset.map(tokenize)

# Load model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=8)

# Training setup
training_args = TrainingArguments(
    output_dir="./model",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_dir="./logs",
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer
)

# Train
trainer.train()

# Save the model and tokenizer
model.save_pretrained("model")
tokenizer.save_pretrained("model")
