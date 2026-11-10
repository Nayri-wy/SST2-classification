import time
import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

label_mapping = {
    "very_negative": 0,
    "negative": 1,
    "neutral": 2,
    "positive": 3,
    "very_positive": 4
}

df = pd.read_csv("sentimentClassification.csv")
df['labels'] = df['sentiment_class'].map(label_mapping)

train_df = df[df["splitset_label"] == 1]
test_df = df[df["splitset_label"] == 2]
dev_df = df[df["splitset_label"] == 3]

# Convert to Hugging Face Dataset format
train_ds = Dataset.from_pandas(train_df[["clean text", "labels"]])
test_ds  = Dataset.from_pandas(test_df[["clean text", "labels"]])
dev_ds   = Dataset.from_pandas(dev_df[["clean text", "labels"]])


# Tokenization

model_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def tokenize_batch(batch):
    return tokenizer(batch["clean text"], truncation=True, max_length=128)

train_ds = train_ds.map(tokenize_batch, batched=True)
test_ds  = test_ds.map(tokenize_batch, batched=True)
dev_ds   = dev_ds.map(tokenize_batch, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "macro_f1": macro_f1}


# Model initilization and training setup

label_names = list(label_mapping.keys())
id2label = {i: name for i, name in enumerate(label_names)}
label2id = {name: i for i, name in enumerate(label_names)}

model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint,
    num_labels=5,
    id2label=id2label,
    label2id=label2id,
).to(device)

training_args = TrainingArguments(
    output_dir="./distilbert-sst5",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    logging_steps=50,
    fp16=torch.cuda.is_available(),  # Enable mixed precision if GPU is available
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=dev_ds,
    #tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Training
train_start = time.perf_counter()
trainer.train()
training_time = time.perf_counter() - train_start

# Inference on test set
infer_start = time.perf_counter()
raw_predictions = trainer.predict(test_ds)
inference_time = time.perf_counter() - infer_start

test_preds = np.argmax(raw_predictions.predictions, axis=-1)
test_labels = raw_predictions.label_ids
latency_per_sample_ms = (inference_time / len(test_labels)) * 1000

test_acc = accuracy_score(test_labels, test_preds)
test_macro_f1 = f1_score(test_labels, test_preds, average="macro")


print("\n" + "=" * 50)
print(f"Training Time:         {training_time:.3f} s")
print(f"Test Inference Time:   {inference_time:.3f} s ({len(test_labels)} samples)")
print(f"Per-sample Latency:    {latency_per_sample_ms:.4f} ms")
print(f"Test Accuracy:         {test_acc * 100:.2f}%")
print(f"Test Macro F1:         {test_macro_f1:.4f}")
print("=" * 50 + "\n")

print(classification_report(test_labels, test_preds, target_names=label_names, digits=4))