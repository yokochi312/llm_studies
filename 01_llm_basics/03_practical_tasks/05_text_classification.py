#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
テキスト分類タスク
このスクリプトでは、従来の機械学習手法と最新のLLMを使用したテキスト分類を実装し、比較します。
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

# 日本語テキスト用のフォント設定
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Yu Gothic', 'Hiragino Sans GB', 'Microsoft YaHei', 'WenQuanYi Micro Hei']

# シード固定
np.random.seed(42)

# サンプルデータ（ニュース記事のカテゴリ分類用）
sample_data = [
    {"text": "日経平均株価は、前日比300円高の30,000円で取引を終えた。米国市場の好調を受けて投資家心理が改善した。", "category": "経済"},
    {"text": "新型スマートフォンの発売日が来月に決定。新機能として高性能カメラと折りたたみディスプレイを搭載。", "category": "テクノロジー"},
    {"text": "サッカー日本代表が国際試合で3-1で勝利。エースストライカーが2ゴールを決める活躍を見せた。", "category": "スポーツ"},
    {"text": "台風10号が週末に九州地方に接近する見込み。気象庁は警戒を呼びかけている。", "category": "天気"},
    {"text": "政府は新たな経済対策を発表。中小企業向けの支援策を拡充する方針。", "category": "政治"},
    {"text": "東京証券取引所の新システムが稼働開始。取引速度が従来の2倍に向上。", "category": "経済"},
    {"text": "新たな人工知能技術が医療診断の精度を向上させることが研究で明らかに。", "category": "テクノロジー"},
    {"text": "プロ野球セ・リーグは首位チームが連勝を7に伸ばし、優勝に向けて前進。", "category": "スポーツ"},
    {"text": "来週は全国的に晴れの日が多く、気温も上昇する見込み。", "category": "天気"},
    {"text": "国会で新たな法案が可決。来年度から施行される予定。", "category": "政治"},
    {"text": "日銀は金融政策の現状維持を決定。市場関係者からは様々な反応。", "category": "経済"},
    {"text": "次世代通信規格の実証実験が都内で開始。通信速度は現行の10倍以上。", "category": "テクノロジー"},
    {"text": "テニスの国際大会で日本人選手が準決勝進出。快進撃に期待が高まる。", "category": "スポーツ"},
    {"text": "梅雨前線の影響で、西日本を中心に大雨の恐れ。土砂災害に警戒。", "category": "天気"},
    {"text": "首相が外国訪問へ。貿易協定の交渉が主な議題となる見込み。", "category": "政治"},
    {"text": "大手自動車メーカーの業績が回復。世界的な半導体不足の影響が緩和。", "category": "経済"},
    {"text": "新たなVR技術が教育分野での活用が進む。没入型学習環境に期待。", "category": "テクノロジー"},
    {"text": "冬季オリンピックのメダル候補が国内大会で新記録を樹立。", "category": "スポーツ"},
    {"text": "今夜から明日にかけて、北日本で雪の予報。交通機関への影響に注意。", "category": "天気"},
    {"text": "地方自治体が新たな観光振興策を発表。コロナ後の経済回復を目指す。", "category": "政治"},
    {"text": "仮想通貨市場が活況。主要銘柄が軒並み価格上昇。", "category": "経済"},
    {"text": "量子コンピュータの研究で日本チームが新たな成果。実用化に前進。", "category": "テクノロジー"},
    {"text": "プロバスケットボールリーグが新シーズン開幕。注目選手の移籍も話題に。", "category": "スポーツ"},
    {"text": "関東地方で花粉の飛散量が増加。例年より早いペースでの飛散開始。", "category": "天気"},
    {"text": "環境問題に関する国際会議が開催。各国の削減目標が議論される見通し。", "category": "政治"},
]

def prepare_data():
    """
    データの準備と前処理を行う関数
    
    Returns:
        tuple: 訓練データと評価データのタプル
    """
    # データフレームの作成
    df = pd.DataFrame(sample_data)
    
    # 訓練データとテストデータに分割
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['category'], test_size=0.3, random_state=42, stratify=df['category']
    )
    
    return X_train, X_test, y_train, y_test

def traditional_ml_classification(X_train, X_test, y_train, y_test):
    """
    従来の機械学習手法によるテキスト分類を行う関数
    
    Args:
        X_train: 訓練テキストデータ
        X_test: テストテキストデータ
        y_train: 訓練ラベルデータ
        y_test: テストラベルデータ
        
    Returns:
        dict: 各モデルの結果を含む辞書
    """
    results = {}
    
    # TF-IDF特徴量の抽出
    vectorizer = TfidfVectorizer(max_features=1000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # 各モデルの訓練と評価
    models = {
        "ロジスティック回帰": LogisticRegression(max_iter=1000),
        "ランダムフォレスト": RandomForestClassifier(),
        "ナイーブベイズ": MultinomialNB()
    }
    
    for name, model in models.items():
        start_time = time.time()
        model.fit(X_train_tfidf, y_train)
        train_time = time.time() - start_time
        
        start_time = time.time()
        y_pred = model.predict(X_test_tfidf)
        inference_time = time.time() - start_time
        
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        results[name] = {
            "accuracy": accuracy,
            "report": report,
            "train_time": train_time,
            "inference_time": inference_time,
            "predictions": y_pred
        }
    
    return results, vectorizer

def llm_classification(X_test, y_test):
    """
    LLMを使用したテキスト分類を行う関数
    
    Args:
        X_test: テストテキストデータ
        y_test: テストラベルデータ
        
    Returns:
        dict: 結果を含む辞書
    """
    try:
        from transformers import pipeline
        
        categories = ["経済", "テクノロジー", "スポーツ", "天気", "政治"]
        
        try:
            # 日本語モデルの使用を試みる
            classifier = pipeline("zero-shot-classification", 
                                 model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
        except:
            # 日本語モデルがなければ多言語モデルを使用
            classifier = pipeline("zero-shot-classification", 
                                 model="facebook/bart-large-mnli")
            print("日本語専用モデルがないため、多言語モデルを使用します（精度が低い可能性があります）")
        
        start_time = time.time()
        predictions = []
        
        for text in X_test:
            try:
                result = classifier(text, categories)
                predicted_category = result['labels'][0]
                predictions.append(predicted_category)
            except Exception as e:
                print(f"予測中にエラーが発生しました: {e}")
                # エラーが発生した場合はランダムなカテゴリを予測
                predictions.append(np.random.choice(categories))
        
        inference_time = time.time() - start_time
        
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, output_dict=True)
        
        return {
            "accuracy": accuracy,
            "report": report,
            "inference_time": inference_time,
            "predictions": predictions
        }
    
    except ImportError:
        print("\n※ LLMによる分類を試すには transformers ライブラリが必要です")
        print("pip install transformers torch を実行してください")
        return None

def predict_new_text(text, traditional_models, vectorizer, categories):
    """
    新しいテキストに対して予測を行う関数
    
    Args:
        text: 予測対象のテキスト
        traditional_models: 訓練済みの従来モデル
        vectorizer: TF-IDFベクトライザ
        categories: カテゴリのリスト
        
    Returns:
        dict: 各モデルの予測結果
    """
    # テキストをベクトル化
    text_vector = vectorizer.transform([text])
    
    results = {}
    
    # 各モデルで予測
    for name, model_info in traditional_models.items():
        model = model_info["model"]
        prediction = model.predict(text_vector)[0]
        probabilities = model.predict_proba(text_vector)[0]
        
        # カテゴリごとの確率を取得
        category_probs = {}
        for i, category in enumerate(model.classes_):
            category_probs[category] = probabilities[i]
        
        results[name] = {
            "prediction": prediction,
            "probabilities": category_probs
        }
    
    # LLMによる予測（可能な場合）
    try:
        from transformers import pipeline
        
        try:
            classifier = pipeline("zero-shot-classification", 
                                 model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
        except:
            classifier = pipeline("zero-shot-classification", 
                                 model="facebook/bart-large-mnli")
        
        result = classifier(text, categories)
        
        results["LLM (Zero-shot)"] = {
            "prediction": result['labels'][0],
            "probabilities": {label: score for label, score in zip(result['labels'], result['scores'])}
        }
    except:
        pass
    
    return results

def main():
    print("===== テキスト分類デモ =====")
    
    # データの準備
    X_train, X_test, y_train, y_test = prepare_data()
    print(f"訓練データ数: {len(X_train)}, テストデータ数: {len(X_test)}")
    print(f"カテゴリ: {', '.join(sorted(set(y_train)))}")
    
    # 従来の機械学習手法による分類
    print("\n----- 従来の機械学習手法 -----")
    traditional_results, vectorizer = traditional_ml_classification(X_train, X_test, y_train, y_test)
    
    for name, result in traditional_results.items():
        print(f"\n{name}:")
        print(f"精度: {result['accuracy']:.4f}")
        print(f"訓練時間: {result['train_time']:.4f}秒")
        print(f"推論時間: {result['inference_time']:.4f}秒")
        
        # モデルを結果に保存（後で新しいテキストの予測に使用）
        traditional_results[name]["model"] = result.pop("model", None)
    
    # LLMによる分類（オプション）
    llm_results = None
    try:
        print("\n----- LLMによるゼロショット分類 (要Transformersライブラリ) -----")
        print("注: 初回実行時はモデルをダウンロードするため時間がかかります")
        
        llm_results = llm_classification(X_test, y_test)
        
        if llm_results:
            print(f"\nLLM (Zero-shot):")
            print(f"精度: {llm_results['accuracy']:.4f}")
            print(f"推論時間: {llm_results['inference_time']:.4f}秒")
    except Exception as e:
        print(f"LLM分類の実行中にエラーが発生しました: {e}")
    
    # 新しいテキストの予測デモ
    print("\n----- 新しいテキストの予測デモ -----")
    new_texts = [
        "株価指数が過去最高値を更新し、投資家からの注目が集まっている。",
        "新しいAIアシスタントが日常生活をサポートする機能を搭載して登場。",
        "選手たちは厳しいトレーニングを重ね、来月の国際大会に備えている。"
    ]
    
    categories = sorted(set(y_train))
    
    for i, text in enumerate(new_texts):
        print(f"\n予測対象テキスト {i+1}: {text}")
        
        try:
            # テキストをベクトル化
            text_vector = vectorizer.transform([text])
            
            # 各モデルで予測
            for name, result in traditional_results.items():
                model = result["model"]
                if model is not None:
                    prediction = model.predict(text_vector)[0]
                    probabilities = model.predict_proba(text_vector)[0]
                    
                    # カテゴリごとの確率を取得
                    category_probs = {}
                    for i, category in enumerate(model.classes_):
                        category_probs[category] = probabilities[i]
                    
                    print(f"\n{name}の予測: {prediction}")
                    print("カテゴリごとの確率:")
                    
                    # 確率の降順でソート
                    sorted_probs = sorted(category_probs.items(), key=lambda x: x[1], reverse=True)
                    for category, prob in sorted_probs:
                        print(f"  - {category}: {prob:.4f}")
            
            # LLMによる予測（可能な場合）
            try:
                from transformers import pipeline
                
                try:
                    classifier = pipeline("zero-shot-classification", 
                                         model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
                except:
                    classifier = pipeline("zero-shot-classification", 
                                         model="facebook/bart-large-mnli")
                
                result = classifier(text, categories)
                
                print(f"\nLLM (Zero-shot)の予測: {result['labels'][0]}")
                print("カテゴリごとの確率:")
                for label, score in zip(result['labels'], result['scores']):
                    print(f"  - {label}: {score:.4f}")
            except Exception as e:
                print(f"LLM予測中にエラーが発生しました: {e}")
        
        except Exception as e:
            print(f"予測中にエラーが発生しました: {e}")

if __name__ == "__main__":
    main() 