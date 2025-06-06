#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
テキスト要約タスク
このスクリプトでは、抽出的要約と生成的要約の両方の手法を実装し、比較します。
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 日本語テキスト用のフォント設定
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Yu Gothic', 'Hiragino Sans GB', 'Microsoft YaHei', 'WenQuanYi Micro Hei']

# シード固定
np.random.seed(42)

# サンプルテキスト
long_text = """
人工知能（AI）技術は現代社会に革命をもたらしています。特に自然言語処理（NLP）の分野では、大規模言語モデル（LLM）の登場により、人間のような文章生成や対話が可能になりました。これらのモデルは、膨大なテキストデータから学習し、文脈を理解する能力を持っています。

GPT（Generative Pre-trained Transformer）シリーズは、その代表例です。GPTモデルは、入力されたテキストの続きを予測するように訓練されており、質問応答、要約、翻訳など多様なタスクをこなすことができます。しかし、幻覚（実際には存在しない情報の生成）や偏見の問題も指摘されています。

一方、BERTなどの双方向エンコーダーモデルは、文脈を前後から理解することに特化しており、テキスト分類や情報抽出などのタスクで高い性能を発揮します。また、T5のようなエンコーダー・デコーダーモデルは、様々な言語タスクを統一的に扱うことができます。

近年では、より少ないデータでの学習を可能にするファインチューニング技術や、モデルサイズを小さくする蒸留技術なども発展しています。また、ユーザーの意図に沿った応答を返すよう人間からのフィードバックを活用する「強化学習」手法も注目されています。

これらの技術の進歩により、AIは医療、教育、ビジネスなど様々な分野での応用が期待されています。例えば医療分野では、患者の症状から可能性のある疾患を提案したり、膨大な医学論文から関連情報を抽出したりするのに役立ちます。

教育分野では、個々の学生のペースや理解度に合わせたパーソナライズされた学習コンテンツの提供が可能になります。ビジネスでは、顧客対応の自動化やデータ分析による意思決定支援など、効率化とコスト削減に貢献しています。

しかしながら、AIの発展には課題も存在します。プライバシーの問題、著作権の問題、雇用への影響など、社会的・倫理的な観点からの検討が必要です。また、エネルギー消費量の増大による環境への影響も無視できません。

将来的には、よりコンパクトで効率的なモデルの開発や、マルチモーダル（テキスト、画像、音声などを統合的に扱う）AIの発展が予想されます。また、AIと人間がより自然に協働できるインターフェースの研究も進むでしょう。

AIの進化は止まることなく続いており、私たちの生活や社会のあり方を大きく変えていく可能性を秘めています。その可能性を最大限に活かしつつ、リスクを最小限に抑えるためのガバナンスの確立が今後の重要な課題となるでしょう。
"""

# 参照用の人間による要約
reference_summary = """
人工知能技術、特に大規模言語モデルの発展により、人間のような文章生成や対話が可能になった。GPTやBERT、T5などの各種モデルは異なる強みを持ち、ファインチューニングや蒸留技術も進化している。これらの技術は医療、教育、ビジネスなど様々な分野で応用が期待される一方、プライバシーや著作権、環境負荷などの課題も存在する。今後はよりコンパクトなモデルやマルチモーダルAIの発展が予想され、AIと人間の協働のためのインターフェース研究も進むだろう。AIの可能性を活かしリスクを抑えるガバナンスの確立が重要な課題となる。
"""

def extractive_summarize(text, num_sentences=5):
    """
    TF-IDFベースの抽出的要約を行う関数
    
    Args:
        text (str): 要約対象のテキスト
        num_sentences (int): 抽出する文の数
        
    Returns:
        str: 抽出された文を結合した要約
    """
    # 文章を分割
    sentences = text.replace('\n', ' ').split('。')
    sentences = [s + '。' for s in sentences if s.strip()]
    
    # TF-IDF行列の作成
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sentences)
    
    # 文章間の類似度を計算
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # 各文の重要度スコアを計算
    scores = np.sum(similarity_matrix, axis=1)
    
    # スコアの高い文のインデックスを取得
    ranked_indices = np.argsort(scores)[::-1]
    top_indices = ranked_indices[:num_sentences]
    top_indices = sorted(top_indices)  # 元の順序を維持
    
    # 選択された文を結合して要約を作成
    summary = ''.join([sentences[i] for i in top_indices])
    
    return summary

def evaluate_rouge(reference, hypothesis):
    """
    簡易的なROUGE-1スコアを計算する関数
    
    Args:
        reference (str): 参照要約
        hypothesis (str): 評価対象の要約
        
    Returns:
        float: ROUGE-1のF1スコア
    """
    # 文字単位で分割
    ref_chars = set(reference)
    hyp_chars = set(hypothesis)
    
    # 共通の文字数を計算
    common_chars = ref_chars.intersection(hyp_chars)
    
    # 適合率、再現率、F1スコアを計算
    precision = len(common_chars) / len(hyp_chars) if len(hyp_chars) > 0 else 0
    recall = len(common_chars) / len(ref_chars) if len(ref_chars) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {"precision": precision, "recall": recall, "f1": f1}

def main():
    print("===== テキスト要約デモ =====")
    print(f"原文の長さ: {len(long_text)} 文字")
    print(f"参照要約の長さ: {len(reference_summary)} 文字")
    
    # 抽出的要約の実行と評価
    print("\n----- 抽出的要約 -----")
    for sentence_count in [3, 5, 7]:
        start_time = time.time()
        summary = extractive_summarize(long_text, num_sentences=sentence_count)
        exec_time = time.time() - start_time
        
        rouge_score = evaluate_rouge(reference_summary, summary)
        
        print(f"\n{sentence_count}文抽出した要約 (長さ: {len(summary)}文字, 実行時間: {exec_time:.3f}秒)")
        print(f"ROUGE-1スコア: P={rouge_score['precision']:.3f}, R={rouge_score['recall']:.3f}, F1={rouge_score['f1']:.3f}")
        print(f"要約: {summary}")
    
    # オプション: Transformersがインストールされている場合は生成的要約も試す
    try:
        from transformers import pipeline
        
        print("\n----- 生成的要約 (要Transformersライブラリ) -----")
        print("注: 初回実行時はモデルをダウンロードするため時間がかかります")
        
        try:
            # 日本語モデルがあればそれを使用
            summarizer = pipeline("summarization", model="sonoisa/t5-base-japanese")
            input_text = "要約: " + long_text
        except:
            # なければ英語モデルを使用（この場合、テキストも英語に変更する必要あり）
            summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
            input_text = long_text
            print("日本語モデルがないため、英語モデルを使用します（正常に動作しない可能性があります）")
        
        start_time = time.time()
        try:
            result = summarizer(input_text, max_length=150, min_length=40, do_sample=False)
            model_summary = result[0]['summary_text']
            exec_time = time.time() - start_time
            
            rouge_score = evaluate_rouge(reference_summary, model_summary)
            
            print(f"\n生成的要約 (長さ: {len(model_summary)}文字, 実行時間: {exec_time:.3f}秒)")
            print(f"ROUGE-1スコア: P={rouge_score['precision']:.3f}, R={rouge_score['recall']:.3f}, F1={rouge_score['f1']:.3f}")
            print(f"要約: {model_summary}")
        except Exception as e:
            print(f"生成的要約の実行中にエラーが発生しました: {e}")
    
    except ImportError:
        print("\n※ 生成的要約を試すには transformers ライブラリが必要です")
        print("pip install transformers sentencepiece を実行してください")

if __name__ == "__main__":
    main() 