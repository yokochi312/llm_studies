"""
シンプルなLLMモデル比較スクリプト

このスクリプトは、異なる言語モデルの基本的な比較を行います。
"""

import torch
from transformers import pipeline
import time
import json
import psutil
import os

def measure_memory():
    """現在のメモリ使用量を測定"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # MB単位

def compare_models(models, prompts):
    """異なるモデルの比較"""
    results = {}
    
    for model_name in models:
        print(f"\n===== モデル: {model_name} =====")
        
        # メモリ使用量（開始時）
        start_memory = measure_memory()
        
        # Text Generation Pipelineの初期化
        start_time = time.time()
        generator = pipeline('text-generation', model=model_name)
        load_time = time.time() - start_time
        
        print(f"モデル読み込み時間: {load_time:.2f}秒")
        
        model_results = {
            "load_time": load_time,
            "prompts": {}
        }
        
        for prompt in prompts:
            print(f"\nプロンプト: {prompt}")
            
            # テキスト生成の実行と時間計測
            start_time = time.time()
            output = generator(prompt, max_length=100, num_return_sequences=1)
            generation_time = time.time() - start_time
            
            generated_text = output[0]['generated_text']
            
            # プロンプトの部分を除去して生成されたテキストのみを表示
            if generated_text.startswith(prompt):
                response_only = generated_text[len(prompt):].strip()
            else:
                response_only = generated_text
            
            model_results["prompts"][prompt] = {
                "full_text": generated_text,
                "response_only": response_only,
                "generation_time": generation_time
            }
            
            print(f"生成されたテキスト: {response_only}")
            print(f"生成時間: {generation_time:.2f}秒")
        
        # メモリ使用量（終了時）
        end_memory = measure_memory()
        memory_usage = end_memory - start_memory
        
        model_results["memory_usage"] = memory_usage
        print(f"メモリ使用量: {memory_usage:.2f}MB")
        
        results[model_name] = model_results
        
        # メモリの解放
        del generator
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return results

def generate_report(results):
    """結果からレポートを生成"""
    report = {
        "summary": {},
        "details": results
    }
    
    # サマリーの作成
    for model_name, model_data in results.items():
        avg_time = 0
        prompt_count = 0
        
        for prompt, data in model_data["prompts"].items():
            avg_time += data["generation_time"]
            prompt_count += 1
        
        if prompt_count > 0:
            avg_time /= prompt_count
        
        report["summary"][model_name] = {
            "average_generation_time": avg_time,
            "load_time": model_data["load_time"],
            "memory_usage": model_data["memory_usage"]
        }
    
    return report

def main():
    # 比較するモデル
    models = [
        "rinna/japanese-gpt2-small",
        "rinna/japanese-gpt2-medium"
    ]
    
    # テスト用のプロンプト
    prompts = [
        "人工知能とは、",
        "データサイエンスの主な目的は、",
        "機械学習の応用例として、"
    ]
    
    # モデルの比較実行
    results = compare_models(models, prompts)
    
    # レポートの生成
    report = generate_report(results)
    
    # 結果をJSONファイルとして保存
    with open("model_comparison_results.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print("\n結果はmodel_comparison_results.jsonに保存されました。")

if __name__ == "__main__":
    main() 