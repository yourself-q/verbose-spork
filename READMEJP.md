# ウェイクワード検出システム

このリポジトリには、ウェイクワード検出のためのデータ収集、前処理、特徴抽出、モデル学習用のスクリプトが含まれています。

## 使用方法

以下の手順でデータ収集とモデルの学習ができます。

### 1. Web UI を起動する

```bash
python app.py
```
Web UI が起動し、ブラウザ上のフォルダに音声を保存できます。

#### データ収集時の注意点

- ウェイクワードを含む音声を録音する場合、周囲の音があっても構いません。むしろ、周囲の音があった方が良いでしょう。
- ウェイクワードを含まない音声を録音する。
- ウェイクワードは人ごとに使用しても構いません。

### 2. 長い音声の分割

```bash
python split_long_audio.py
```

`split_long_audio.py` を実行すると、長い音声録音が2秒ごとに自動的に分割されます。環境音の録音時間は何分でも構いません。

### 3. 特徴抽出

```bash
python extract_features.py
```

`extract_features.py` を実行して音声データから特徴を抽出し、トレーニングデータを使いやすい `.pkl` ファイルに変換します。

### 4. データセットの準備

```bash
python prepare_dataset.py
```

`prepare_dataset.py` を実行して `.pkl` ファイルをトレーニングデータと検証データに分割します。

> **注意**
> 実行時に次のエラーが表示された場合は、
> ```text
> ValueError: 配列要素にシーケンスを設定しています。要求された配列は、2次元後に不均一な形状になります。検出された形状は (14, 40) + 不均一な部分です。
> ```
> 事前に `split_long_audio.py` を実行していない可能性があります。もう一度実行してください。

### 5. モデルのトレーニング

```bash
python train_cnn.py
```

`train_cnn.py` は軽量 CNN モデルを定義し、トレーニングします。出力モデルファイルには日付が付けられるため、バージョン管理が容易です。

### 6. 推論

```bash
python predict.py --model models/wakeword_cnn_<timestamp>.pth --audio path/to/audio.wav
```

`predict.py` を実行すると、トレーニング済みモデルを使用して WAV ファイルで推論が実行され、予測ラベルと信頼度が出力されます。

## ファイル一覧

- [app.py](./app.py)
- [split_long_audio.py](./split_long_audio.py)
- [extract_features.py](./extract_features.py)
- [prepare_dataset.py](./prepare_dataset.py)
- [train_cnn.py](./train_cnn.py)
- [predict.py](./predict.py)

## Web UIによる学習と推論

### Web UIを起動

\\`\\\`\\\`bash
python app.py
\\\`\\\`\\\`

ブラウザで http://localhost:5000 を開きます。

- **モデル学習** ボタンをクリックすると、\`run.sh\` による学習が開始され、ログがリアルタイムで表示されます。
- **推論実行** では、モデルを選択し、音声ファイルをアップロードして「推論実行」をクリックします。予測ラベルと信頼度が表示されます。


