#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
コンテンツ生成タスク
このスクリプトでは、言語モデルを使用して様々な形式のテキストを生成する方法を実装します。
"""

import time
import random
import numpy as np

# シード固定
np.random.seed(42)
random.seed(42)

# テンプレートベースの生成に使用するデータ
templates = {
    "製品紹介": [
        "{製品名}は、{特徴}が特徴の{製品カテゴリ}です。{ユーザー層}に最適で、{利点}を実現します。価格は{価格}で、{購入場所}でお求めいただけます。",
        "新発売の{製品名}をご紹介します。この{製品カテゴリ}は{特徴}が魅力で、{利点}を可能にします。特に{ユーザー層}におすすめです。{価格}という価格も魅力の一つです。{購入場所}にてお買い求めください。",
        "{ユーザー層}待望の{製品カテゴリ}、{製品名}が登場しました。{特徴}を備え、{利点}を実現します。{価格}でご購入いただけるこの製品は、{購入場所}で発売中です。"
    ],
    "レストランレビュー": [
        "{レストラン名}に行ってきました。{場所}にあるこのお店は、{料理タイプ}が評判です。特に{おすすめメニュー}は絶品でした。{価格帯}で、{雰囲気}な雰囲気が特徴です。{サービス}なサービスも魅力的でした。{総合評価}です。",
        "{場所}にある{レストラン名}を訪れました。{料理タイプ}のお店で、{おすすめメニュー}を注文しました。味は{総合評価}で、{価格帯}という価格も納得です。店内は{雰囲気}で、スタッフの対応も{サービス}でした。",
        "{レストラン名}のレビューです。{場所}にあり、{料理タイプ}を提供しています。私のイチオシは{おすすめメニュー}で、{総合評価}と言えます。店内の{雰囲気}な雰囲気と{サービス}なサービスも印象的でした。価格帯は{価格帯}です。"
    ],
    "旅行記": [
        "{場所}への旅行記です。{季節}に訪れ、{期間}滞在しました。{観光スポット}や{アクティビティ}を楽しみました。{宿泊施設}に宿泊し、{食事}を堪能しました。{天候}でしたが、全体的には{総合評価}な旅でした。",
        "{季節}の{場所}へ{期間}の旅に出かけました。{宿泊施設}に滞在し、{観光スポット}を訪問しました。また、{アクティビティ}も体験しました。{食事}が印象的で、{天候}の中でしたが、{総合評価}と感じました。",
        "{期間}かけて{場所}を旅行しました。{季節}だったので、{天候}でした。{宿泊施設}に泊まり、{観光スポット}を巡りました。{アクティビティ}も楽しく、{食事}も美味しかったです。総じて{総合評価}な旅でした。"
    ]
}

# テンプレート変数の候補
variables = {
    # 製品紹介用
    "製品名": ["SmartX Pro", "EcoLight", "PowerMax 3000", "UltraSlim Book", "ClearView Monitor", "QuietCool Fan", "FitTrack Watch", "SoundPods", "CleanAir Purifier", "TastyChef Cooker"],
    "製品カテゴリ": ["スマートフォン", "照明器具", "パワーバンク", "ノートパソコン", "モニター", "扇風機", "スマートウォッチ", "ワイヤレスイヤホン", "空気清浄機", "調理器具"],
    "特徴": ["高性能プロセッサ", "省エネ設計", "大容量バッテリー", "軽量コンパクト", "高解像度", "静音設計", "健康管理機能", "ノイズキャンセリング", "HEPAフィルター", "多機能"],
    "ユーザー層": ["ビジネスパーソン", "学生", "アウトドア愛好家", "クリエイター", "ゲーマー", "高齢者", "スポーツ愛好家", "音楽ファン", "アレルギー持ちの方", "料理愛好家"],
    "利点": ["作業効率の向上", "電気代の削減", "外出先での充電の心配がなく", "持ち運びの負担を軽減", "目の疲れを軽減", "静かな環境の確保", "健康状態の把握", "没入感のある音楽体験", "きれいな空気の確保", "調理時間の短縮"],
    "価格": ["39,800円", "5,980円", "12,000円", "89,800円", "35,000円", "15,800円", "25,000円", "18,500円", "32,000円", "28,500円"],
    "購入場所": ["当社オンラインストア", "全国の家電量販店", "Amazonなどのオンラインマーケット", "当社直営店舗", "提携小売店", "限定予約サイト", "公式アプリ", "専門店", "百貨店", "公式SNSリンク"],
    
    # レストランレビュー用
    "レストラン名": ["イル・グスト", "和心", "ブラッセリー・ポム", "スパイスガーデン", "オーシャンテラス", "モダン食堂", "ビストロ・シェルブルー", "炭火焼肉 匠", "カフェ・ソレイユ", "麺屋 一期一会"],
    "場所": ["駅前", "閑静な住宅街", "ショッピングモール内", "オフィス街", "海沿い", "公園の近く", "古い商店街", "高層ビルの最上階", "観光地のメインストリート", "隠れ家的な路地"],
    "料理タイプ": ["イタリアン", "和食", "フレンチ", "インド料理", "シーフード", "創作料理", "ビストロ料理", "焼肉", "カフェ料理", "ラーメン"],
    "おすすめメニュー": ["トリュフパスタ", "季節の刺身盛り合わせ", "鴨のコンフィ", "バターチキンカレー", "シーフードプラッター", "シェフの気まぐれコース", "牛ほほ肉の赤ワイン煮込み", "特選和牛盛り合わせ", "有機野菜のサラダ", "特製濃厚豚骨ラーメン"],
    "価格帯": ["リーズナブル", "やや高め", "コスパ良好", "高級", "学生でも安心", "ランチはお得", "ディナーは予算必要", "値段以上の価値あり", "少し贅沢な価格", "庶民的"],
    "雰囲気": ["落ち着いた", "活気ある", "おしゃれ", "アットホーム", "開放的", "シック", "カジュアル", "伝統的", "モダン", "ロマンチック"],
    "サービス": ["丁寧", "気さく", "テキパキとした", "少し物足りない", "期待以上", "親身になってくれる", "知識豊富", "笑顔の素敵な", "控えめながら行き届いた", "フレンドリーな"],
    "総合評価": ["大満足", "また訪れたい", "期待通り", "少し期待はずれ", "驚きの美味しさ", "コストパフォーマンス抜群", "特別な日におすすめ", "日常使いにぴったり", "一度は訪れる価値あり", "地元の隠れた名店"],
    
    # 旅行記用
    "季節": ["春", "夏", "秋", "冬", "桜の季節", "紅葉シーズン", "雪景色の", "新緑の", "梅雨の", "連休中の"],
    "期間": ["2泊3日", "日帰り", "1週間", "3泊4日", "長期滞在", "週末", "5日間", "10日間", "半日", "4泊5日"],
    "観光スポット": ["有名な寺院", "美術館", "歴史的な城", "国立公園", "地元の市場", "展望台", "世界遺産", "隠れた名所", "人気のビーチ", "古い町並み"],
    "アクティビティ": ["ハイキング", "地元料理の料理教室", "サイクリング", "クルーズ", "ショッピング", "温泉巡り", "スキー", "シュノーケリング", "文化体験", "ワインテイスティング"],
    "宿泊施設": ["高級ホテル", "ゲストハウス", "伝統的な旅館", "民泊", "リゾートホテル", "ペンション", "キャンプ場", "カプセルホテル", "古民家ステイ", "ビジネスホテル"],
    "食事": ["地元の郷土料理", "海鮮料理", "ミシュラン星付きレストラン", "屋台グルメ", "伝統的な朝食", "ビュッフェ", "有名店のスイーツ", "ファーマーズマーケットの新鮮食材", "地ビール", "地元のワイン"],
    "天候": ["晴天", "小雨", "曇り時々晴れ", "予想以上の悪天候", "穏やかな気候", "蒸し暑い", "肌寒い", "絶好の観光日和", "風が強かった", "天気に恵まれた"],
    "総合評価": ["最高", "忘れられない", "またぜひ訪れたい", "期待以上", "少し期待外れ", "思い出深い", "充実した", "癒された", "冒険的な", "学びの多い"]
}

def fill_template(template_type, variables_dict):
    """
    テンプレートを変数で埋めて文章を生成する関数
    
    Args:
        template_type (str): テンプレートの種類
        variables_dict (dict): 変数の候補辞書
        
    Returns:
        str: 生成されたテキスト
    """
    if template_type not in templates:
        return "指定されたテンプレートタイプが見つかりません。"
    
    # テンプレートをランダムに選択
    template = random.choice(templates[template_type])
    
    # 必要な変数を抽出
    needed_variables = []
    current_var = ""
    in_variable = False
    
    for char in template:
        if char == '{':
            in_variable = True
            current_var = ""
        elif char == '}' and in_variable:
            in_variable = False
            needed_variables.append(current_var)
        elif in_variable:
            current_var += char
    
    # 変数を実際の値に置換
    result = template
    for var_name in needed_variables:
        if var_name in variables_dict:
            value = random.choice(variables_dict[var_name])
            result = result.replace("{" + var_name + "}", value)
    
    return result

def markov_chain_generator(training_texts, order=2, length=100):
    """
    マルコフ連鎖を使用してテキストを生成する関数
    
    Args:
        training_texts (list): 学習用テキストのリスト
        order (int): マルコフ連鎖の次数
        length (int): 生成するテキストの最大長さ
        
    Returns:
        str: 生成されたテキスト
    """
    # モデルの構築（状態→次の文字の辞書）
    model = {}
    
    for text in training_texts:
        # パディング
        padded_text = "^" * order + text
        
        for i in range(len(padded_text) - order):
            state = padded_text[i:i+order]
            next_char = padded_text[i+order]
            
            if state not in model:
                model[state] = []
            model[state].append(next_char)
    
    # テキストの生成
    current_state = "^" * order
    result = ""
    
    for _ in range(length):
        if current_state not in model:
            break
        
        next_char = random.choice(model[current_state])
        result += next_char
        current_state = current_state[1:] + next_char
    
    return result

# サンプルの学習テキスト
sample_texts = [
    "人工知能技術の発展は、私たちの生活を大きく変えています。特に自然言語処理の分野では、機械が人間の言語を理解し、生成する能力が飛躍的に向上しています。",
    "大規模言語モデルは、膨大なテキストデータから学習し、驚くほど自然な文章を生成できるようになりました。しかし、その精度や倫理的な側面にはまだ課題も残されています。",
    "機械学習の基本的な仕組みは、データからパターンを抽出し、そのパターンに基づいて予測や分類を行うことです。深層学習はその一種で、ニューラルネットワークを用いて複雑なパターンを学習します。",
    "自然言語処理技術の応用範囲は広く、翻訳、要約、質問応答、感情分析など多岐にわたります。これらの技術は、ビジネス、医療、教育など様々な分野で活用されています。",
    "プログラミング言語には様々な種類があり、用途によって適したものが異なります。例えば、Pythonはデータ分析やAI開発に、JavaScriptはWeb開発に広く使用されています。"
]

def evaluate_quality(generated_text, original=None):
    """
    生成されたテキストの品質を評価する簡易的な関数
    
    Args:
        generated_text (str): 生成されたテキスト
        original (str, optional): 比較元のテキスト（ある場合）
        
    Returns:
        dict: 評価結果
    """
    # 文章の長さ
    length = len(generated_text)
    
    # 文の数
    sentences = generated_text.split('。')
    sentence_count = sum(1 for s in sentences if s.strip())
    
    # 平均文長
    avg_sentence_length = length / sentence_count if sentence_count > 0 else 0
    
    # 語彙の多様性（ユニークな文字の割合）
    unique_chars = len(set(generated_text))
    char_diversity = unique_chars / length if length > 0 else 0
    
    # 結果
    result = {
        "length": length,
        "sentence_count": sentence_count,
        "avg_sentence_length": avg_sentence_length,
        "char_diversity": char_diversity
    }
    
    # オリジナルとの比較（ある場合）
    if original:
        # 文字の共通性
        common_chars = set(generated_text).intersection(set(original))
        char_overlap = len(common_chars) / len(set(original)) if original else 0
        
        result["char_overlap"] = char_overlap
    
    return result

def main():
    print("===== コンテンツ生成デモ =====")
    
    # 1. テンプレートベースの生成
    print("\n----- テンプレートベースの生成 -----")
    for template_type in templates.keys():
        start_time = time.time()
        generated_text = fill_template(template_type, variables)
        generation_time = time.time() - start_time
        
        print(f"\n{template_type}の生成例 (生成時間: {generation_time:.3f}秒):")
        print(generated_text)
        
        # 品質評価
        quality = evaluate_quality(generated_text)
        print(f"文字数: {quality['length']}, 文数: {quality['sentence_count']}, 平均文長: {quality['avg_sentence_length']:.1f}, 語彙多様性: {quality['char_diversity']:.3f}")
    
    # 2. マルコフ連鎖による生成
    print("\n----- マルコフ連鎖による生成 -----")
    
    for order in [1, 2, 3]:
        start_time = time.time()
        generated_text = markov_chain_generator(sample_texts, order=order, length=200)
        generation_time = time.time() - start_time
        
        print(f"\n次数{order}のマルコフ連鎖による生成例 (生成時間: {generation_time:.3f}秒):")
        print(generated_text)
        
        # 品質評価
        quality = evaluate_quality(generated_text, sample_texts[0])
        print(f"文字数: {quality['length']}, 文数: {quality['sentence_count']}, 平均文長: {quality['avg_sentence_length']:.1f}, 語彙多様性: {quality['char_diversity']:.3f}, 共通性: {quality.get('char_overlap', 0):.3f}")
    
    # 3. プロンプトベースの生成（Transformersライブラリが必要）
    try:
        from transformers import pipeline
        
        print("\n----- 高度なモデルによる生成 (要Transformersライブラリ) -----")
        print("注: 初回実行時はモデルをダウンロードするため時間がかかります")
        
        try:
            # 日本語生成モデルを使用
            generator = pipeline("text-generation", model="rinna/japanese-gpt2-medium")
            prompt_text = "人工知能は今後、私たちの生活を"
        except:
            # なければ英語モデルを使用
            generator = pipeline("text-generation")
            prompt_text = "Artificial intelligence will"
            print("日本語モデルがないため、英語モデルを使用します")
        
        # インタラクティブモード
        print("\nプロンプトを入力してテキスト生成を試してみましょう（終了するには 'exit' と入力）")
        print("例: '人工知能は今後、私たちの生活を'")
        
        while True:
            user_prompt = input("\nプロンプト: ")
            if user_prompt.lower() == 'exit':
                break
            
            try:
                start_time = time.time()
                result = generator(user_prompt, max_length=100, num_return_sequences=1)
                generation_time = time.time() - start_time
                
                generated_text = result[0]["generated_text"]
                print(f"\n生成結果 (実行時間: {generation_time:.3f}秒):")
                print(generated_text)
                
                # 品質評価
                quality = evaluate_quality(generated_text, user_prompt)
                print(f"文字数: {quality['length']}, 文数: {quality['sentence_count']}, 平均文長: {quality['avg_sentence_length']:.1f}, 語彙多様性: {quality['char_diversity']:.3f}")
            except Exception as e:
                print(f"生成中にエラーが発生しました: {e}")
    
    except ImportError:
        print("\n※ 高度なテキスト生成を試すには transformers ライブラリが必要です")
        print("pip install transformers を実行してください")

if __name__ == "__main__":
    main() 