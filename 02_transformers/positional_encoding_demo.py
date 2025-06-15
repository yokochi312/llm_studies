"""
位置エンコーディングの仕組みを具体的な数値で理解するためのデモ
"""
import numpy as np

def simple_positional_encoding(seq_len, d_model):
    """簡単な位置エンコーディングの実装"""
    # 位置エンコーディング行列を初期化
    pos_enc = np.zeros((seq_len, d_model))
    
    # 各位置と各次元に対して値を計算
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            # 偶数インデックスには sin を使用
            pos_enc[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
            
            # 奇数インデックスには cos を使用（範囲内であれば）
            if i + 1 < d_model:
                pos_enc[pos, i + 1] = np.cos(pos / (10000 ** (i / d_model)))
    
    return pos_enc

def demonstrate_positional_encoding():
    """位置エンコーディングのデモンストレーション"""
    print("=" * 80)
    print("位置エンコーディングのデモンストレーション")
    print("=" * 80)
    
    # 小さなサイズで位置エンコーディングを計算
    seq_len = 5  # 5つの単語
    d_model = 8  # 8次元の埋め込み
    
    print(f"\n1. 位置エンコーディングの計算 (シーケンス長={seq_len}, 埋め込み次元={d_model})")
    print("-" * 60)
    
    pos_enc = simple_positional_encoding(seq_len, d_model)
    
    # 各位置の位置エンコーディングを表示
    for pos in range(seq_len):
        print(f"\n位置 {pos} の位置エンコーディング:")
        print(pos_enc[pos])
    
    # 特定の例で詳細に説明
    print("\n\n2. 位置エンコーディングの計算例")
    print("-" * 60)
    
    pos = 2  # 位置2（3番目の単語）
    dim = 0  # 次元0（最初の次元）
    
    value = np.sin(pos / (10000 ** (dim / d_model)))
    print(f"位置 {pos}, 次元 {dim} (sin): {value:.6f}")
    print(f"計算式: sin({pos} / (10000 ^ ({dim} / {d_model})))")
    print(f"      = sin({pos} / (10000 ^ {dim/d_model:.6f}))")
    print(f"      = sin({pos} / {10000 ** (dim/d_model):.6f})")
    print(f"      = sin({pos / (10000 ** (dim/d_model)):.6f})")
    print(f"      = {value:.6f}")
    
    dim = 1  # 次元1（2番目の次元）
    value = np.cos(pos / (10000 ** (dim / d_model)))
    print(f"\n位置 {pos}, 次元 {dim} (cos): {value:.6f}")
    print(f"計算式: cos({pos} / (10000 ^ ({dim} / {d_model})))")
    print(f"      = cos({pos} / (10000 ^ {dim/d_model:.6f}))")
    print(f"      = cos({pos} / {10000 ** (dim/d_model):.6f})")
    print(f"      = cos({pos / (10000 ** (dim/d_model)):.6f})")
    print(f"      = {value:.6f}")
    
    # 異なる位置の単語の比較
    print("\n\n3. 異なる位置の単語の比較")
    print("-" * 60)
    
    pos1, pos2 = 0, 1  # 隣接する位置
    print(f"位置 {pos1} と位置 {pos2} の位置エンコーディングの比較（隣接する位置）:")
    print(f"位置 {pos1}: {pos_enc[pos1]}")
    print(f"位置 {pos2}: {pos_enc[pos2]}")
    print(f"差分: {pos_enc[pos2] - pos_enc[pos1]}")
    print(f"ユークリッド距離: {np.linalg.norm(pos_enc[pos2] - pos_enc[pos1]):.6f}")
    
    pos1, pos2 = 0, 4  # 離れた位置
    print(f"\n位置 {pos1} と位置 {pos2} の位置エンコーディングの比較（離れた位置）:")
    print(f"位置 {pos1}: {pos_enc[pos1]}")
    print(f"位置 {pos2}: {pos_enc[pos2]}")
    print(f"差分: {pos_enc[pos2] - pos_enc[pos1]}")
    print(f"ユークリッド距離: {np.linalg.norm(pos_enc[pos2] - pos_enc[pos1]):.6f}")
    
    # 実際の単語ベクトルに位置エンコーディングを加える例
    print("\n\n4. 単語ベクトルに位置エンコーディングを加える例")
    print("-" * 60)
    
    # 簡単な単語ベクトルの例（実際にはもっと複雑）
    word_vectors = {
        "猫": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
        "犬": np.array([0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    }
    
    # 「猫が犬を追いかけた」の例
    sentence1 = ["猫", "が", "犬", "を", "追いかけた"]
    # 「犬が猫を追いかけた」の例
    sentence2 = ["犬", "が", "猫", "を", "追いかけた"]
    
    # 「猫」と「犬」の単語ベクトルを表示
    print("元の単語ベクトル:")
    print(f"「猫」: {word_vectors['猫']}")
    print(f"「犬」: {word_vectors['犬']}")
    
    # 「猫」が文の先頭にある場合と3番目にある場合の比較
    cat_pos0 = word_vectors["猫"] + pos_enc[0]  # 文の先頭の「猫」
    cat_pos2 = word_vectors["猫"] + pos_enc[2]  # 文の3番目の「猫」
    
    print("\n位置情報を加えた単語ベクトル:")
    print(f"「猫」(位置0): {cat_pos0}")
    print(f"「猫」(位置2): {cat_pos2}")
    print(f"差分: {cat_pos2 - cat_pos0}")
    
    # 同じ文中の「猫」と「犬」の比較
    print("\n同じ文中の異なる単語の比較:")
    print(f"「猫」(位置0): {cat_pos0}")
    dog_pos2 = word_vectors["犬"] + pos_enc[2]  # 文の3番目の「犬」
    print(f"「犬」(位置2): {dog_pos2}")
    
    # 要約
    print("\n\n5. まとめ")
    print("-" * 60)
    print("位置エンコーディングの特徴:")
    print("1. 各位置に固有のパターンを与える")
    print("2. 近い位置は似たパターンを持つ")
    print("3. 遠い位置は異なるパターンを持つ")
    print("4. 単語の意味ベクトルに位置情報を追加することで、Transformerは単語の順序を理解できる")
    print("5. 同じ単語でも文中の位置が異なれば、異なるベクトル表現になる")

if __name__ == "__main__":
    demonstrate_positional_encoding() 