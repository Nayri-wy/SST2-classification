import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report

label_mapping = {
    "very_negative": 0,
    "negative": 1,
    "neutral": 2,
    "positive": 3,
    "very_positive": 4
}

df = pd.read_csv("sentimentClassification.csv")
df['sentiment_label'] = df['sentiment_class'].map(label_mapping)

train_df = df[df["splitset_label"] == 1]
test_df = df[df["splitset_label"] == 2]
dev_df = df[df["splitset_label"] == 3]

X_train, y_train = train_df["clean text"], train_df["sentiment_label"]
X_test, y_test = test_df["clean text"], test_df["sentiment_label"]


# TF-IDF + Logistic Regression Pipeline

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=50000)),
    ("clf", LogisticRegression(max_iter=1000, C=2.0, solver="lbfgs"))
])

t0 = time.perf_counter()
pipeline.fit(X_train, y_train)
train_time = time.perf_counter() - t0

t1 = time.perf_counter()
y_test_pred = pipeline.predict(X_test)
infer_time = time.perf_counter() - t1
latency_ms = (infer_time / len(X_test)) * 1000


# Evaluation metrics

acc = accuracy_score(y_test, y_test_pred)
macro_f1 = f1_score(y_test, y_test_pred, average="macro")

print(f"--- Benchmark Results ---")
print(f"Training Time:         {train_time:.3f} s")
print(f"Test Inference Time:   {infer_time:.3f} s ({len(X_test)} samples)")
print(f"Per-sample Latency:    {latency_ms:.4f} ms")
print(f"Test Accuracy:         {acc * 100:.2f}%")
print(f"Test Macro F1:         {macro_f1:.4f}")

classes = list(label_mapping.keys())
print(classification_report(y_test, y_test_pred, target_names=classes, digits=4))
