import tkinter as tk
import sounddevice as sd
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

def record_audio():
    duration = 5  # Recording duration in seconds
    sample_rate = 22050  # Sampling rate
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    return audio, sample_rate

def extract_mfcc(audio, sr):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    return mfccs

def extract_spectrogram(audio, sr):
    D = librosa.stft(audio)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    return S_db

def display_features():
    audio, sample_rate = record_audio()
    mfccs = extract_mfcc(audio[:, 0], sample_rate)
    spectrogram = extract_spectrogram(audio[:, 0], sample_rate)

    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    librosa.display.specshow(spectrogram, sr=sample_rate, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')

    plt.subplot(2, 1, 2)
    librosa.display.specshow(mfccs, sr=sample_rate, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')

    plt.tight_layout()
    plt.show()

# GUI Setup
root = tk.Tk()
root.title("Feature Extraction")

extract_button = tk.Button(root, text="Extract Features", command=display_features)
extract_button.pack()

root.mainloop()
