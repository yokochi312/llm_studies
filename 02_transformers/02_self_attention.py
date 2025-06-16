"""
セルフアテンションのメカニズムを理解するためのデモ
"""
import numpy as np

def softmax(x):
    """ソフトマックス関数の実装"""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def self_attention_demo(query, key, value):
    """
    セルフアテンションの計算過程を示すデモ関数
    
    引数:
        query: クエリベクトル [seq_len, d_k]
        key: キーベクトル [seq_len, d_k]
        value: バリューベクトル [seq_len, d_v]
    
    戻り値:
        attention_output: アテンション出力 [seq_len, d_v]
    """
    # スケーリング係数
    d_k = key.shape[1]
    scaling_factor = np.sqrt(d_k)
    
    # アテンションスコアの計算: Q * K^T / sqrt(d_k)
    attention_scores = np.matmul(query, key.T) / scaling_factor
    
    # ソフトマックスでアテンション重みに変換
    attention_weights = softmax(attention_scores)
    
    # 最終的なアテンション出力を計算: weights * V
    attention_output = np.matmul(attention_weights, value)
    
    return attention_output, attention_weights

def demonstrate_self_attention():
    """セルフアテンションのデモンストレーション"""
    print("=" * 80)
    print("セルフアテンションのデモンストレーション")
    print("=" * 80)
    
    # 簡単な例：3単語の文章、各単語は4次元のベクトルで表現
    seq_len = 3
    d_k = 4  # クエリとキーの次元
    d_v = 4  # バリューの次元
    
    # 単純化のため、ランダムではなく意図的な値を設定
    # 単語ベクトルの例（実際の埋め込みはもっと高次元）
    word_embeddings = np.array([
        [1.0, 0.2, 0.3, 0.1],  # 単語1 "猫"
        [0.2, 1.0, 0.1, 0.3],  # 単語2 "は"
        [0.3, 0.1, 1.0, 0.2]   # 単語3 "寝る"
    ])
    
    print("\n1. 入力単語埋め込み:")
    print("-" * 60)
    words = ["猫", "は", "寝る"]
    for i, word in enumerate(words):
        print(f"単語 '{word}': {word_embeddings[i]}")
    
    # 線形変換行列（実際にはニューラルネットワークで学習される）
    # 簡単のため単位行列に近い値を使用
    W_q = np.eye(d_k) + np.random.normal(0, 0.1, (d_k, d_k))
    W_k = np.eye(d_k) + np.random.normal(0, 0.1, (d_k, d_k))
    W_v = np.eye(d_v) + np.random.normal(0, 0.1, (d_v, d_v))
    
    print("\n2. 線形変換行列:")
    print("-" * 60)
    print(f"W_q (クエリ変換行列):\n{W_q.round(3)}")
    print(f"W_k (キー変換行列):\n{W_k.round(3)}")
    print(f"W_v (バリュー変換行列):\n{W_v.round(3)}")
    
    # Q, K, Vベクトルの計算
    Q = np.matmul(word_embeddings, W_q)
    K = np.matmul(word_embeddings, W_k)
    V = np.matmul(word_embeddings, W_v)
    
    print("\n3. Q, K, Vベクトル:")
    print("-" * 60)
    for i, word in enumerate(words):
        print(f"単語 '{word}':")
        print(f"  Q: {Q[i].round(3)}")
        print(f"  K: {K[i].round(3)}")
        print(f"  V: {V[i].round(3)}")
    
    # セルフアテンションの計算
    attention_output, attention_weights = self_attention_demo(Q, K, V)
    
    print("\n4. アテンションスコアとアテンション重み:")
    print("-" * 60)
    # スケーリング係数
    scaling_factor = np.sqrt(d_k)
    # 生のアテンションスコア
    raw_scores = np.matmul(Q, K.T)
    # スケーリングされたスコア
    scaled_scores = raw_scores / scaling_factor
    
    print("生のアテンションスコア (Q * K^T):")
    for i, word in enumerate(words):
        scores_str = " ".join([f"{score:.3f}" for score in raw_scores[i]])
        print(f"  '{word}' -> [{scores_str}]")
    
    print(f"\nスケーリングされたアテンションスコア (Q * K^T / sqrt({d_k})):")
    for i, word in enumerate(words):
        scores_str = " ".join([f"{score:.3f}" for score in scaled_scores[i]])
        print(f"  '{word}' -> [{scores_str}]")
    
    print("\nアテンション重み (softmax後):")
    for i, word in enumerate(words):
        weights_str = " ".join([f"{weight:.3f}" for weight in attention_weights[i]])
        print(f"  '{word}' -> [{weights_str}]")
    
    print("\n5. 最終的なアテンション出力:")
    print("-" * 60)
    for i, word in enumerate(words):
        print(f"単語 '{word}' の出力: {attention_output[i].round(3)}")
    
    # 各単語がどの単語に注目しているかを視覚化
    print("\n6. アテンションの解釈:")
    print("-" * 60)
    for i, word in enumerate(words):
        print(f"単語 '{word}' のアテンション分布:")
        for j, target in enumerate(words):
            weight = attention_weights[i, j]
            bar = "#" * int(weight * 50)
            print(f"  -> '{target}': {weight:.3f} {bar}")
    
    # まとめ
    print("\n7. セルフアテンションのまとめ:")
    print("-" * 60)
    print("1. 各単語はクエリ(Q)、キー(K)、バリュー(V)ベクトルに変換される")
    print("2. 各単語のクエリと全ての単語のキーの内積でアテンションスコアを計算")
    print("3. スケーリング後、ソフトマックスでアテンション重みに変換")
    print("4. アテンション重みと各単語のバリューベクトルの加重和が出力")
    print("5. これにより各単語は文脈に応じて他の単語の情報を取り込める")
    print("6. マルチヘッドアテンションでは、この処理を複数回並列で行う")
    print("7. セルフアテンションにより、Transformerは長距離依存関係を効率的に捉えられる")

if __name__ == "__main__":
    demonstrate_self_attention() 