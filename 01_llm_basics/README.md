# LLMの基礎

## 概要
このセクションでは、大規模言語モデル（LLM）の基礎的な概念から、その仕組みまでを学習します。

## 学習目標
- LLMの基本的な概念を理解する
- 言語モデルの発展の歴史を学ぶ
- 代表的なLLMの特徴を把握する
- トークン化の仕組みを理解する

## コンテンツ

### 1. LLMの基本概念 (`01_introduction/`)
- LLMとは何か
- なぜLLMが重要なのか
- LLMの主な用途と応用分野
- LLMの限界と課題

### 2. 言語モデルの歴史 (`02_history/`)
- 統計的言語モデル
- Word2VecからBERTまで
- GPTシリーズの進化
- 最新のLLMの動向

### 3. 主要なLLMの種類と特徴 (`03_models/`)
- GPT-3/4の特徴
- Claude
- LLaMA
- BERT/RoBERTa
- オープンソースLLM

### 4. トークン化とエンコーディング (`04_tokenization/`)
- トークン化の基本
- BPE (Byte Pair Encoding)
- WordPiece
- SentencePiece
- 実践的なトークン化の演習

### 5. 実践演習 (`05_practice/`)
- Python環境のセットアップ
- 簡単なトークン化の実装
- HuggingFaceライブラリの基本的な使い方
- トークン数のカウントと最適化

## 必要な環境
- Python 3.8以上
- pip or conda
- 基本的なPythonの知識

## セットアップ手順
```bash
# 仮想環境の作成
python -m venv venv
source venv/bin/activate  # Linuxの場合
# または
.\venv\Scripts\activate  # Windowsの場合

# 必要なパッケージのインストール
pip install -r requirements.txt
```

## 学習の進め方
1. 各ディレクトリ内のREADMEを順番に読む
2. サンプルコードを実行して動作を確認
3. 演習問題に取り組む
4. 理解度チェックの質問に答える

## 参考資料
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer論文
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/index)
- [OpenAI GPT-3 Paper](https://arxiv.org/abs/2005.14165) 