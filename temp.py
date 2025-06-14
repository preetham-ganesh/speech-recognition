import librosa

import time

import numpy as np


def audio_to_features(filepath: str, n_mels: int = 161) -> np.ndarray:
    # Load audio (LibriSpeech is 16kHz)
    y, sr = librosa.load(filepath, sr=16000)

    # Compute log-mel spectrogram
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

    # Transpose: librosa returns (freq_bins, time_steps), but we want (time_steps, freq_bins)
    log_spectrogram = log_spectrogram.T
    return log_spectrogram


start_time = time.time()
ls = audio_to_features(
    "data/extracted_data/librispeech/train/LibriSpeech/train-clean-360/957/132568/957-132568-0028.flac"
)
print(ls.shape)

print(type(ls))
print(round(time.time() - start_time, 3))

with open("temp.npy", "wb") as f:
    np.save(f, ls)


start_time = time.time()
temp = np.load("temp.npy")
print(round(time.time() - start_time, 7))
