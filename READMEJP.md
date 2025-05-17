# ウェイクワード検出システム

このリポジトリでは、ウェイクワード検出のためのデータ収集、前処理、特徴量抽出、モデル学習を行うスクリプトをまとめています。

## 使用方法

以下の手順でデータの収集からモデル学習までを行うことができます。

### 1. Web UIの起動

```bash
python app.py
```
これで Web UI が起動し、ブラウザ上で音声をフォルダー単位で保存できます。

#### データ収集のポイント

- ウェイクワードを含む音声を録音する際、環境音があっても問題ありません。むしろ環境音があるほうが良いです。
- ウェイクワードを含まない音声も録音してください。
- 人ごとにウェイクワードを分けても問題ありません。

### 2. 長時間音声の分割

```bash
python split_long_audio.py
```

`split_long_audio.py` を実行すると、長時間録音した音声を 2 秒ごとに自動で分割します。環境音録音が何分であっても問題ありません。

### 3. 特徴量抽出

```bash
python extract_features.py
```

`extract_features.py` を実行して、音声データから特徴量を抽出し、学習用データを扱いやすい `.pkl` ファイルに変換します。

### 4. データセット準備

```bash
python prepare_dataset.py
```

`prepare_dataset.py` を実行して `.pkl` ファイルを学習用データと検証用データに分割します。

> **注意**  
> 実行時に以下のようなエラーが表示された場合は、  
> ```text
> ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 2 dimensions. The detected shape was (14, 40) + inhomogeneous part.
> ```  
> 事前に `split_long_audio.py` を実行していない可能性があります。再度実行してください。

### 5. モデルの学習

```bash
python train_cnn.py
```

`train_cnn.py` では軽量な CNN モデルを定義し、学習を行います。出力されるモデルファイルには日付が付加されるため、バージョン管理がしやすくなっています。

## ファイル一覧

- [app.py](./app.py)
- [split_long_audio.py](./split_long_audio.py)
- [extract_features.py](./extract_features.py)
- [prepare_dataset.py](./prepare_dataset.py)
- [train_cnn.py](./train_cnn.py)

