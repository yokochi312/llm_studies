#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
感情分析タスク
このスクリプトでは、テキストの感情（ポジティブ/ネガティブ/中立）を分析する手法を実装します。
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# 日本語テキスト用のフォント設定
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Yu Gothic', 'Hiragino Sans GB', 'Microsoft YaHei', 'WenQuanYi Micro Hei']

# シード固定
np.random.seed(42)

# サンプルデータセット（日本語のレビューとその感情ラベル）
sample_reviews = [
    # ポジティブなレビュー (1)
    "この商品は素晴らしいです。使いやすく、高品質で大満足しています。",
    "期待以上の商品でした。これからもずっと使い続けたいです。",
    "素晴らしい買い物ができて幸せです。友人にも強くおすすめします。",
    "長年愛用していますが、壊れることなく使えています。信頼性が高いです。",
    "この価格でこの品質は驚きです。コストパフォーマンスが非常に高いと感じました。",
    "操作が直感的で初心者でも簡単に使えます。デザインも洗練されていて気に入っています。",
    "問い合わせに対する対応も迅速で丁寧でした。安心して利用できるサービスだと思います。",
    "配送も早く、梱包も丁寧でした。また利用したいと思います。",
    "機能が豊富で、日々の作業が格段に効率化されました。もっと早く知りたかったです。",
    "子供も喜んで使っています。家族全員のお気に入りの商品です。",
    
    # ネガティブなレビュー (0)
    "商品が届くまでに3週間もかかりました。対応の遅さに失望しています。",
    "説明書が分かりにくく、使い方を理解するのに苦労しました。改善を希望します。",
    "値段の割に品質が悪いです。すぐに壊れてしまいました。",
    "カスタマーサポートに問い合わせても回答が得られませんでした。",
    "商品の色が写真と全く違いました。非常に残念です。",
    "サイズが合わず、返品しようとしましたが手続きが複雑で困りました。",
    "動作が不安定で、頻繁にエラーが発生します。使い物になりません。",
    "注文した商品と異なるものが届きました。確認が不十分だったようです。",
    "梱包が粗雑で、商品が破損した状態で届きました。ショックです。",
    "初期不良があり、交換を依頼しましたが対応されませんでした。二度と買いません。",
    
    # 中立的なレビュー (2)
    "普通の商品です。特に良くも悪くもありません。",
    "値段相応の商品だと思います。特に不満はありませんが、感動もありません。",
    "他の商品と比較していないので、良し悪しの判断が難しいです。",
    "商品自体は問題ないですが、もう少し機能が充実していると良かったです。",
    "想像通りの商品でした。普通に使えています。",
    "まだ使い始めたばかりなので、長期的な評価はこれからです。",
    "人によって合う合わないがありそうな商品です。私は普通に使えています。",
    "機能は満足していますが、デザインはもう少し改善の余地があると思います。",
    "価格を考えれば妥当な商品だと思います。大きな不満はありません。",
    "特徴的な機能はありませんが、基本的な機能は備えています。必要十分です。"
]

# 感情ラベル（1: ポジティブ, 0: ネガティブ, 2: 中立）
sample_labels = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

# 感情ラベルを文字列に変換する辞書
label_names = {1: "ポジティブ", 0: "ネガティブ", 2: "中立"}

def train_simple_classifier():
    """
    単純なBag-of-Words + ナイーブベイズ分類器を訓練する関数
    
    Returns:
        tuple: 訓練済みモデル、ベクトライザー、評価結果
    """
    # データとラベルを訓練用とテスト用に分割
    X_train, X_test, y_train, y_test = train_test_split(
        sample_reviews, sample_labels, test_size=0.3, random_state=42
    )
    
    # Bag-of-Wordsベクトル化
    vectorizer = CountVectorizer(token_pattern=r'(?u)\b\w+\b', min_df=1)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # ナイーブベイズ分類器の訓練
    classifier = MultinomialNB()
    classifier.fit(X_train_vec, y_train)
    
    # 予測と評価
    y_pred = classifier.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    
    # 分類レポート
    report = classification_report(y_test, y_pred, target_names=[label_names[0], label_names[1], label_names[2]])
    
    # 混同行列
    cm = confusion_matrix(y_test, y_pred)
    
    results = {
        "accuracy": accuracy,
        "report": report,
        "confusion_matrix": cm,
        "test_data": X_test,
        "test_labels": y_test,
        "predicted_labels": y_pred
    }
    
    return classifier, vectorizer, results

def predict_sentiment(text, classifier, vectorizer):
    """
    テキストの感情を予測する関数
    
    Args:
        text (str): 分析対象のテキスト
        classifier: 訓練済み分類器
        vectorizer: 訓練済みベクトライザー
        
    Returns:
        dict: 予測結果（ラベルと確率）
    """
    # テキストをベクトル化
    text_vec = vectorizer.transform([text])
    
    # 予測
    label = classifier.predict(text_vec)[0]
    probabilities = classifier.predict_proba(text_vec)[0]
    
    return {
        "sentiment": label,
        "sentiment_name": label_names[label],
        "probabilities": {label_names[i]: prob for i, prob in enumerate(probabilities) if i in label_names}
    }

def visualize_results(results):
    """
    分類結果を可視化する関数
    
    Args:
        results (dict): 分類結果
    """
    # 2x2のサブプロットを作成
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 混同行列のヒートマップ
    cm = results["confusion_matrix"]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[label_names[0], label_names[1], label_names[2]],
                yticklabels=[label_names[0], label_names[1], label_names[2]], ax=ax1)
    ax1.set_title("混同行列")
    ax1.set_xlabel("予測ラベル")
    ax1.set_ylabel("実際のラベル")
    
    # 2. 感情分布の円グラフ
    label_counts = pd.Series(results["test_labels"]).value_counts().sort_index()
    ax2.pie(label_counts, labels=[label_names[i] for i in label_counts.index], 
            autopct='%1.1f%%', startangle=90, colors=["#ff9999", "#66b3ff", "#99ff99"])
    ax2.set_title("テストデータの感情分布")
    
    # 3. 予測の正確さの棒グラフ
    correct = results["predicted_labels"] == results["test_labels"]
    accuracy_by_class = {}
    for i in range(len(results["test_labels"])):
        label = results["test_labels"][i]
        if label not in accuracy_by_class:
            accuracy_by_class[label] = {"correct": 0, "total": 0}
        accuracy_by_class[label]["total"] += 1
        if correct[i]:
            accuracy_by_class[label]["correct"] += 1
    
    class_accuracy = {label: data["correct"] / data["total"] for label, data in accuracy_by_class.items()}
    ax3.bar([label_names[label] for label in class_accuracy.keys()], list(class_accuracy.values()), color=["#ff9999", "#66b3ff", "#99ff99"])
    ax3.set_title("クラスごとの予測精度")
    ax3.set_ylim(0, 1)
    ax3.set_ylabel("正解率")
    
    # 4. 誤分類サンプルのテーブル
    incorrect_samples = []
    for i in range(len(results["test_labels"])):
        if not correct[i]:
            incorrect_samples.append({
                "テキスト": results["test_data"][i],
                "実際": label_names[results["test_labels"][i]],
                "予測": label_names[results["predicted_labels"][i]]
            })
    
    if incorrect_samples:
        ax4.axis('off')
        table_data = [[sample["テキスト"][:50] + "..." if len(sample["テキスト"]) > 50 else sample["テキスト"], 
                        sample["実際"], sample["予測"]] for sample in incorrect_samples[:5]]
        table = ax4.table(cellText=table_data, colLabels=["テキスト", "実際", "予測"], loc='center', cellLoc='left')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        ax4.set_title("誤分類サンプル (最大5件)")
    else:
        ax4.text(0.5, 0.5, "誤分類サンプルなし", ha='center', va='center', fontsize=12)
        ax4.axis('off')
        ax4.set_title("誤分類サンプル")
    
    plt.tight_layout()
    plt.savefig("sentiment_analysis_results.png")
    plt.show()

def main():
    print("===== 感情分析デモ =====")
    print(f"サンプルデータ数: {len(sample_reviews)}")
    
    # 分類器の訓練
    print("\n----- 分類器の訓練 -----")
    start_time = time.time()
    classifier, vectorizer, results = train_simple_classifier()
    training_time = time.time() - start_time
    
    print(f"訓練時間: {training_time:.3f}秒")
    print(f"精度: {results['accuracy']:.3f}")
    print("\n分類レポート:")
    print(results["report"])
    
    # グラフの表示（可能な環境であれば）
    try:
        visualize_results(results)
        print("\n結果のグラフを表示しました。")
    except Exception as e:
        print(f"\nグラフの表示中にエラーが発生しました: {e}")
        print("グラフを表示せずに続行します。")
    
    # インタラクティブモード
    print("\n----- インタラクティブモード -----")
    print("テキストを入力すると感情分析を行います（終了するには 'exit' と入力）")
    
    while True:
        user_text = input("\nテキスト: ")
        if user_text.lower() == 'exit':
            break
        
        start_time = time.time()
        result = predict_sentiment(user_text, classifier, vectorizer)
        predict_time = time.time() - start_time
        
        print(f"感情: {result['sentiment_name']} (実行時間: {predict_time:.3f}秒)")
        print("確率:")
        for sentiment, prob in result["probabilities"].items():
            print(f"  {sentiment}: {prob:.3f}")
    
    # オプション: Transformersがインストールされている場合は高度な感情分析も試す
    try:
        from transformers import pipeline
        
        print("\n----- 高度な感情分析 (要Transformersライブラリ) -----")
        print("注: 初回実行時はモデルをダウンロードするため時間がかかります")
        
        try:
            # 日本語感情分析モデルを使用
            sentiment_analyzer = pipeline("sentiment-analysis", model="daigo/bert-base-japanese-sentiment")
        except:
            # なければ英語モデルを使用
            sentiment_analyzer = pipeline("sentiment-analysis")
            print("日本語モデルがないため、英語モデルを使用します（正常に動作しない可能性があります）")
        
        print("\nモデルベースの感情分析を試してみましょう（終了するには 'exit' と入力）")
        
        while True:
            user_text = input("\nテキスト: ")
            if user_text.lower() == 'exit':
                break
            
            try:
                start_time = time.time()
                result = sentiment_analyzer(user_text)
                exec_time = time.time() - start_time
                
                print(f"感情: {result[0]['label']} (スコア: {result[0]['score']:.3f}, 実行時間: {exec_time:.3f}秒)")
            except Exception as e:
                print(f"エラーが発生しました: {e}")
    
    except ImportError:
        print("\n※ 高度な感情分析を試すには transformers ライブラリが必要です")
        print("pip install transformers を実行してください")

if __name__ == "__main__":
    main() 