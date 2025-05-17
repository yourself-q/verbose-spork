import os
import librosa
import soundfile as sf

SOURCE_DIR = "dataset/environmental"
OUTPUT_DIR = "dataset/environmental"
SEGMENT_LENGTH = 2.0  # 秒
TARGET_SR = 16000

def split_audio(file_path, output_prefix):
    y, sr = librosa.load(file_path, sr=TARGET_SR)
    total_duration = librosa.get_duration(y=y, sr=sr)
    samples_per_segment = int(TARGET_SR * SEGMENT_LENGTH)

    num_segments = int(len(y) / samples_per_segment)

    for i in range(num_segments):
        start_sample = i * samples_per_segment
        end_sample = start_sample + samples_per_segment
        segment = y[start_sample:end_sample]
        output_file = os.path.join(OUTPUT_DIR, f"{output_prefix}_part_{i}.wav")
        sf.write(output_file, segment, TARGET_SR)
        print(f"✅ Saved {output_file}")

def process_folder():
    for file in os.listdir(SOURCE_DIR):
        if not file.endswith(".wav"):
            continue
        filename_wo_ext = os.path.splitext(file)[0]
        full_path = os.path.join(SOURCE_DIR, file)
        split_audio(full_path, filename_wo_ext)

if __name__ == "__main__":
    process_folder()

