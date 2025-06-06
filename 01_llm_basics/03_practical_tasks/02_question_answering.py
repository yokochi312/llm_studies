#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
質問応答システム
このスクリプトでは、与えられた文脈から質問に回答する基本的なシステムを実装します。
"""

import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# サンプルコンテキスト
context = """
人工知能（AI）の歴史は1950年代に始まります。コンピュータサイエンティストのアラン・チューリングは1950年に「機械は考えることができるか」という問いを立て、後に「チューリングテスト」として知られる概念を提案しました。

1956年には、ダートマス会議でジョン・マッカーシーが「人工知能」という用語を初めて使用しました。1960年代から70年代にかけては、シンボリックAIの時代と呼ばれ、ルールベースの専門家システムなどが研究されました。

1980年代には、機械学習の概念が広まり始め、決定木やニューラルネットワークなどの手法が開発されました。しかし、計算能力の制限やデータ不足により「AIの冬」と呼ばれる停滞期も経験しています。

2000年代に入ると、コンピュータの処理能力の向上とインターネットの普及によるデータ量の増加から、機械学習、特にディープラーニングが大きく進展しました。2012年にはアレックス・クリジェフスキーらのチームがディープラーニングを使用して画像認識コンペティションで大幅な精度向上を達成し、AIブームが再燃しました。

2010年代後半には、自然言語処理の分野で転移学習やトランスフォーマーアーキテクチャが登場し、GPTやBERTなどの大規模言語モデルが開発されました。これらのモデルは、文章生成、翻訳、質問応答など多様なタスクで人間に近い性能を示しています。

現在のAI研究は、マルチモーダル学習、強化学習、自己教師あり学習など様々な方向に進んでおり、医療診断、自動運転、クリエイティブ支援など、社会の多くの側面で応用が進んでいます。
"""

# サンプル質問と回答のペア
qa_pairs = [
    {
        "question": "アラン・チューリングは何年に機械は考えることができるかという問いを立てましたか？",
        "answer": "1950年"
    },
    {
        "question": "「人工知能」という用語を初めて使用したのは誰ですか？",
        "answer": "ジョン・マッカーシー"
    },
    {
        "question": "AIの冬とは何ですか？",
        "answer": "計算能力の制限やデータ不足により研究が停滞した時期"
    },
    {
        "question": "2012年に画像認識コンペティションで大幅な精度向上を達成したのは誰ですか？",
        "answer": "アレックス・クリジェフスキーらのチーム"
    },
    {
        "question": "現在のAI研究はどのような方向に進んでいますか？",
        "answer": "マルチモーダル学習、強化学習、自己教師あり学習など様々な方向"
    }
]

def simple_qa_system(question, context):
    """
    TF-IDFと類似度計算を用いた簡易的な質問応答システム
    
    Args:
        question (str): ユーザーからの質問
        context (str): 回答の検索対象となるコンテキスト
        
    Returns:
        str: 回答と考えられる文
    """
    # コンテキストを文に分割
    sentences = context.replace('\n', ' ').split('。')
    sentences = [s + '。' for s in sentences if s.strip()]
    
    # 質問とコンテキストの文をベクトル化
    vectorizer = TfidfVectorizer()
    # 質問も含めてベクトル化
    all_texts = sentences + [question]
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    # 質問と各文の類似度を計算
    question_vector = tfidf_matrix[-1]  # 最後のベクトルが質問
    context_vectors = tfidf_matrix[:-1]  # 残りはコンテキストの文
    similarities = cosine_similarity(question_vector, context_vectors)[0]
    
    # 最も類似度の高い文のインデックスを取得
    most_similar_idx = np.argmax(similarities)
    
    return {
        "answer": sentences[most_similar_idx],
        "confidence": similarities[most_similar_idx],
        "context": sentences[most_similar_idx - 1] if most_similar_idx > 0 else "" + 
                   sentences[most_similar_idx] + 
                   sentences[most_similar_idx + 1] if most_similar_idx < len(sentences) - 1 else ""
    }

def evaluate_answer(predicted, actual):
    """
    予測された回答と実際の回答を評価する関数
    
    Args:
        predicted (str): システムが生成した回答
        actual (str): 正解の回答
        
    Returns:
        float: 回答の正確さの指標（0〜1）
    """
    # 単純な文字列一致率で評価
    pred_words = set(predicted.lower().split())
    actual_words = set(actual.lower().split())
    
    if len(pred_words) == 0 or len(actual_words) == 0:
        return 0.0
    
    common_words = pred_words.intersection(actual_words)
    precision = len(common_words) / len(pred_words)
    recall = len(common_words) / len(actual_words)
    
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return f1

def main():
    print("===== 質問応答システムデモ =====")
    print(f"コンテキスト長: {len(context)} 文字\n")
    
    # サンプル質問で評価
    print("----- サンプル質問の評価 -----")
    total_score = 0
    
    for i, qa in enumerate(qa_pairs):
        question = qa["question"]
        actual_answer = qa["answer"]
        
        start_time = time.time()
        result = simple_qa_system(question, context)
        exec_time = time.time() - start_time
        
        score = evaluate_answer(result["answer"], actual_answer)
        total_score += score
        
        print(f"\n質問 {i+1}: {question}")
        print(f"予測: {result['answer']} (信頼度: {result['confidence']:.3f}, 実行時間: {exec_time:.3f}秒)")
        print(f"正解: {actual_answer}")
        print(f"スコア: {score:.3f}")
    
    print(f"\n平均スコア: {total_score / len(qa_pairs):.3f}")
    
    # インタラクティブモード
    print("\n----- インタラクティブモード -----")
    print("質問を入力してください（終了するには 'exit' と入力）")
    
    while True:
        user_question = input("\n質問: ")
        if user_question.lower() == 'exit':
            break
        
        result = simple_qa_system(user_question, context)
        print(f"回答: {result['answer']} (信頼度: {result['confidence']:.3f})")
        print(f"コンテキスト: {result['context']}")
    
    # オプション: Transformersがインストールされている場合は高度なQAも試す
    try:
        from transformers import pipeline
        
        print("\n----- 高度な質問応答 (要Transformersライブラリ) -----")
        print("注: 初回実行時はモデルをダウンロードするため時間がかかります")
        
        try:
            # 日本語QAモデルを使用
            qa_pipeline = pipeline("question-answering", model="tohoku-nlp/bert-base-japanese-v3")
        except:
            # なければ英語モデルを使用
            qa_pipeline = pipeline("question-answering")
            print("日本語モデルがないため、英語モデルを使用します（正常に動作しない可能性があります）")
        
        print("\nモデルベースの質問応答を試してみましょう（終了するには 'exit' と入力）")
        
        while True:
            user_question = input("\n質問: ")
            if user_question.lower() == 'exit':
                break
            
            try:
                start_time = time.time()
                result = qa_pipeline(question=user_question, context=context)
                exec_time = time.time() - start_time
                
                print(f"回答: {result['answer']} (スコア: {result['score']:.3f}, 実行時間: {exec_time:.3f}秒)")
                print(f"開始位置: {result['start']}, 終了位置: {result['end']}")
            except Exception as e:
                print(f"エラーが発生しました: {e}")
    
    except ImportError:
        print("\n※ 高度な質問応答を試すには transformers ライブラリが必要です")
        print("pip install transformers を実行してください")

if __name__ == "__main__":
    main() 