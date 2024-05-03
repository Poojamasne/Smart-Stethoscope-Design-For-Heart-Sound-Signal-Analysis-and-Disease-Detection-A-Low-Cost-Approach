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

def extract_mfcc(audio, sr):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return mfccs

def extract_zero_crossing_rate(audio):
    zcr = librosa.feature.zero_crossing_rate(y=audio)
    return zcr

def extract_spectral_centroid(audio, sr):
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    return centroid

def plot_features(mfccs, zcr, centroid):
    plt.figure(figsize=(10, 6))
    
    plt.subplot(3, 1, 1)
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title('MFCCs')
    
    plt.subplot(3, 1, 2)
    plt.plot(zcr[0])
    plt.title('Zero Crossing Rate')
    
    plt.subplot(3, 1, 3)
    plt.plot(centroid[0])
    plt.title('Spectral Centroid')

    plt.tight_layout()
    plt.show()

def execute_features():
    audio, sample_rate = record_audio()
    mfccs = extract_mfcc(audio, sample_rate)
    zcr = extract_zero_crossing_rate(audio)
    centroid = extract_spectral_centroid(audio, sample_rate)
    plot_features(mfccs, zcr, centroid)

# GUI Setup
root = tk.Tk()
root.title("Feature Extraction from Heart Sound")

extract_button = tk.Button(root, text="Extract Features", command=execute_features)
extract_button.pack()

root.mainloop()
