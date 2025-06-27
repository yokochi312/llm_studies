import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import evaluate

# 1. データセットの読み込み
print("データセットを読み込んでいます...")
dataset = load_dataset("llm-book/livedoor-news-corpus")

# データセットのラベルを確認
labels = dataset["train"].features["label"].names
id2label = {i: label for i, label in enumerate(labels)}
label2id = {label: i for i, label in enumerate(labels)}
print(f"ラベル: {labels}")


# 2. モデルとトークナイザーの準備
model_name = "cl-tohoku/bert-base-japanese-whole-word-masking"
print(f"モデル '{model_name}' を読み込んでいます...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=len(labels), id2label=id2label, label2id=label2id
)

# デバイスの確認
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)
print(f"使用デバイス: {device}")


# 3. データの前処理
def preprocess_function(examples):
    # テキストをトークナイズ
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

print("データセットを前処理しています...")
tokenized_dataset = dataset.map(preprocess_function, batched=True)


# 4. 評価指標の定義
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy_metric.compute(predictions=predictions, references=labels)


# 5. トレーニングの実行
# トレーニング引数の設定
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1, # 時間短縮のため1エポックに設定
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    load_best_model_at_end=True,
    report_to="none", # wandbなどのレポートを無効化
)

# サブセットで実行（動作確認のため）
small_train_dataset = tokenized_dataset["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_dataset["validation"].shuffle(seed=42).select(range(200))


# Trainerの初期化
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

print("ファインチューニングを開始します...")
trainer.train()

print("ファインチューニングが完了しました。")

# 6. 結果の評価
print("テストデータで評価します...")
# テストデータも少量に
small_test_dataset = tokenized_dataset["test"].shuffle(seed=42).select(range(200))
eval_results = trainer.evaluate(eval_dataset=small_test_dataset)

print(f"テストデータでの評価結果: {eval_results}") 