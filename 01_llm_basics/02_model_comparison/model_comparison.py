"""
LLMモデル比較スクリプト

このスクリプトでは、異なるLLMモデルの特徴と性能を比較します。
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, T5Tokenizer
import time

def load_model(model_name):
    """モデルとトークナイザーを読み込む"""
    print(f"Loading {model_name}...")
    
    # 日本語対応のためにT5トークナイザーを使用
    if "gpt" in model_name.lower():
        tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt2-medium")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer

def generate_text(model, tokenizer, prompt, max_length=100):
    """テキスト生成を行う"""
    start_time = time.time()
    
    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 入力の準備
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 生成設定
    generation_config = {
        "max_length": max_length,
        "num_return_sequences": 1,
        "no_repeat_ngram_size": 3,
        "pad_token_id": tokenizer.pad_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": True,
        "top_k": 40,
        "top_p": 0.9,
        "temperature": 0.9,
        "repetition_penalty": 1.2,
        "length_penalty": 1.0,
    }
    
    # テキスト生成
    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_config)
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    end_time = time.time()
    
    # GPUメモリの解放
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return generated_text, end_time - start_time

def compare_models(models, prompt):
    """複数のモデルを比較する"""
    results = {}
    
    for model_name in models:
        model, tokenizer = load_model(model_name)
        text, generation_time = generate_text(model, tokenizer, prompt)
        
        results[model_name] = {
            "generated_text": text,
            "generation_time": generation_time
        }
        
        print(f"\nResults for {model_name}:")
        print(f"Generated text: {text}")
        print(f"Generation time: {generation_time:.2f} seconds")
        
        # モデルのメモリ解放
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results

def main():
    # テスト用の設定
    models_to_compare = ["rinna/japanese-gpt2-medium"]
    test_prompts = [
        "人工知能は",
        "未来のテクノロジーは",
        "データサイエンスの重要性は"
    ]
    
    print("=== モデル比較開始 ===")
    for prompt in test_prompts:
        print(f"\nテスト プロンプト: {prompt}")
        results = compare_models(models_to_compare, prompt)
    print("\n=== 比較完了 ===")

if __name__ == "__main__":
    main() 