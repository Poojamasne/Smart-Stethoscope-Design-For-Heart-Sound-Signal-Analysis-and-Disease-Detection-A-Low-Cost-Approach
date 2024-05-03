import tkinter as tk
import sounddevice as sd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

def record_audio():
    duration = 5  # Recording duration in seconds
    sample_rate = 44100  # Sampling rate
    channels = 1  # Mono audio

    print("Recording... Press Ctrl+C to stop.")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, dtype='float32')
    sd.wait()
    audio = audio.flatten()
    return audio, sample_rate

def extract_features(audio, sr):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    zcr = librosa.feature.zero_crossing_rate(y=audio)
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr)

    return mfccs, zcr, centroid, bandwidth, contrast, mel_spectrogram

def plot_features(features):
    mfccs, zcr, centroid, bandwidth, contrast, mel_spectrogram = features

    plt.figure(figsize=(12, 10))

    plt.subplot(3, 2, 1)
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title('MFCCs')

    plt.subplot(3, 2, 2)
    plt.plot(zcr[0])
    plt.title('Zero Crossing Rate')

    plt.subplot(3, 2, 3)
    plt.plot(centroid[0])
    plt.title('Spectral Centroid')

    plt.subplot(3, 2, 4)
    plt.plot(bandwidth[0])
    plt.title('Spectral Bandwidth')

    plt.subplot(3, 2, 5)
    librosa.display.specshow(contrast, x_axis='time')
    plt.colorbar()
    plt.title('Spectral Contrast')

    plt.subplot(3, 2, 6)
    librosa.display.specshow(mel_spectrogram, x_axis='time')
    plt.colorbar()
    plt.title('Mel-frequency spectrogram')

    plt.tight_layout()
    plt.show()

def execute_features():
    audio, sample_rate = record_audio()
    features = extract_features(audio, sample_rate)
    plot_features(features)

# GUI Setup
root = tk.Tk()
root.title("Feature Extraction from Heart Sound")

extract_button = tk.Button(root, text="Extract Features", command=execute_features)
extract_button.pack()

root.mainloop()
