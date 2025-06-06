"""
異なるモデルアーキテクチャ比較スクリプト

このスクリプトは、GPT、BERT、T5などの異なるアーキテクチャのモデルを比較します。
"""

import torch
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
import time
import json
import psutil
import os
from typing import Dict, List, Any

# モデルタイプの定義
MODEL_TYPES = {
    "gpt": {
        "model_class": AutoModelForCausalLM,
        "task": "テキスト生成（自己回帰型）",
        "description": "次の単語を予測する自己回帰モデル"
    },
    "bert": {
        "model_class": AutoModelForMaskedLM,
        "task": "マスク言語モデル",
        "description": "マスクされた単語を予測する双方向エンコーダー"
    },
    "t5": {
        "model_class": AutoModelForSeq2SeqLM,
        "task": "シーケンス変換",
        "description": "入力テキストを出力テキストに変換するエンコーダー・デコーダーモデル"
    }
}

def measure_memory():
    """現在のメモリ使用量を測定"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # MB単位

def load_model(model_name: str, model_type: str):
    """モデルとトークナイザーをロード"""
    model_class = MODEL_TYPES[model_type]["model_class"]
    
    start_time = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = model_class.from_pretrained(model_name)
    load_time = time.time() - start_time
    
    return model, tokenizer, load_time

def get_model_size(model):
    """モデルのパラメータ数を計算"""
    return sum(p.numel() for p in model.parameters())

def evaluate_model(model_name: str, model_type: str, test_input: str) -> Dict[str, Any]:
    """モデルの評価"""
    print(f"\n===== モデル: {model_name} ({model_type}) =====")
    
    # メモリ使用量（開始時）
    start_memory = measure_memory()
    
    # モデルのロード
    model, tokenizer, load_time = load_model(model_name, model_type)
    print(f"モデル読み込み時間: {load_time:.2f}秒")
    
    # モデルのパラメータ数
    param_count = get_model_size(model)
    print(f"パラメータ数: {param_count:,}")
    
    # 推論時間の計測
    start_time = time.time()
    
    if model_type == "gpt":
        # GPTモデルの場合はテキスト生成
        inputs = tokenizer(test_input, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_length=50,
                num_return_sequences=1
            )
        result_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    elif model_type == "bert":
        # BERTモデルの場合はマスク予測
        masked_text = test_input.replace("知能", "[MASK]")
        inputs = tokenizer(masked_text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        
        mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
        mask_token_logits = outputs.logits[0, mask_token_index, :]
        top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()
        result_text = [tokenizer.decode([token]) for token in top_5_tokens]
    
    elif model_type == "t5":
        # T5モデルの場合は翻訳や要約
        inputs = tokenizer("summarize: " + test_input, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(inputs.input_ids, max_length=50)
        result_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    inference_time = time.time() - start_time
    print(f"推論時間: {inference_time:.2f}秒")
    
    # メモリ使用量（終了時）
    end_memory = measure_memory()
    memory_usage = end_memory - start_memory
    print(f"メモリ使用量: {memory_usage:.2f}MB")
    
    # 結果の表示
    print(f"入力: {test_input}")
    print(f"出力: {result_text}")
    
    # 結果の保存
    result = {
        "model_name": model_name,
        "model_type": model_type,
        "task": MODEL_TYPES[model_type]["task"],
        "description": MODEL_TYPES[model_type]["description"],
        "load_time": load_time,
        "inference_time": inference_time,
        "memory_usage": memory_usage,
        "parameter_count": param_count,
        "input": test_input,
        "output": result_text
    }
    
    # メモリの解放
    del model, tokenizer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return result

def main():
    # 評価するモデルのリスト（モデル名とタイプのペア）
    models_to_evaluate = [
        ("rinna/japanese-gpt2-small", "gpt"),
        ("cl-tohoku/bert-base-japanese", "bert"),
        ("sonoisa/t5-base-japanese", "t5")
    ]
    
    # テスト入力
    test_input = "人工知能は将来、どのように社会を変えるでしょうか？"
    
    # 結果保存用
    results = []
    
    # 各モデルの評価
    for model_name, model_type in models_to_evaluate:
        try:
            result = evaluate_model(model_name, model_type, test_input)
            results.append(result)
        except Exception as e:
            print(f"エラー: {model_name} の評価中に問題が発生しました: {e}")
    
    # 結果をJSONファイルとして保存
    with open("architecture_comparison_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n結果はarchitecture_comparison_results.jsonに保存されました。")

if __name__ == "__main__":
    main() 