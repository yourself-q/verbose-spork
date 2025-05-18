日本語版のREADME.mdは[こちら](./READMEJP.md)
# Wake word detection system

This repository contains scripts for data collection, preprocessing, feature extraction, and model training for wake word detection.

## How to use

The following procedure can be used to collect data and train the model.

### 1. Start the Web UI

```bash
python app.py
```
This will start the Web UI, and you can save audio in folders on your browser.

#### Points to remember when collecting data

- When recording audio that includes the wake word, it is okay to have ambient sounds. It is better to have ambient sounds.
- Record audio that does not include the wake word.
- It is okay to use separate wake words for each person.

### 2. Splitting long audio

```bash
python split_long_audio.py
```

Running `split_long_audio.py` will automatically split long audio recordings every 2 seconds. It doesn't matter how many minutes the ambient sound recording is.

### 3. Feature Extraction

```bash
python extract_features.py
```

Run `extract_features.py` to extract features from the audio data and convert the training data to an easy-to-use `.pkl` file.

### 4. Dataset Preparation

```bash
python prepare_dataset.py
```

Run `prepare_dataset.py` to split the `.pkl` file into training data and validation data.

> **Note**
> If the following error is displayed when executing,
> ```text
> ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 2 dimensions. The detected shape was (14, 40) + inhomogeneous part.
> ```
> You may not have run `split_long_audio.py` beforehand. Please run it again.

### 5. Model training

```bash
python train_cnn.py
```

`train_cnn.py` defines a lightweight CNN model and trains it. The output model file is date-stamped, making version control easy.

### 6. Inference

```bash
python predict.py --model models/wakeword_cnn_<timestamp>.pth --audio path/to/audio.wav
```

Run `predict.py` to perform inference on a WAV file using a trained model, outputting the predicted label and confidence.

## List of files

- [app.py](./app.py)
- [split_long_audio.py](./split_long_audio.py)
- [extract_features.py](./extract_features.py)
- [prepare_dataset.py](./prepare_dataset.py)
- [train_cnn.py](./train_cnn.py)
- [predict.py](./predict.py)


## Web UI Training and Inference

### Start the Web UI

\\`\\\`\\\`bash
python app.py
\\\`\\\`\\\`

Open your browser at http://localhost:5000 to access the UI.

- **モデル学習**: Click the "モデル学習" button to start training. Logs will stream live in the UI.
- **推論実行**: Select a trained model and upload a WAV or WebM audio file, then click "推論実行" to perform inference. The predicted label and confidence will be displayed.


