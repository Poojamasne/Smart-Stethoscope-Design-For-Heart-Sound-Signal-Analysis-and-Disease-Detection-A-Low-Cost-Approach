import tkinter as tk
import sounddevice as sd
import soundfile as sf
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

def record_audio():
    duration = 5  # Recording duration in seconds
    sample_rate = 44100  # Sampling rate
    channels = 1  # Mono audio

    print("Recording... Press Ctrl+C to stop.")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, dtype='float32')
    sd.wait()
    sf.write('recorded_heart_sound.wav', audio, sample_rate)
    print("Recording saved as 'recorded_heart_sound.wav'")

def plot_ecg_wave():
    audio_file = 'recorded_heart_sound.wav'
    audio, sr = librosa.load(audio_file, sr=None)

    # Processing steps (preemphasis, feature extraction, etc.) can be performed here

    # Identify approximate locations of S1 and S2 (these are arbitrary, replace with actual detection methods)
    s1_location = 0.3  # Example location for S1
    s2_location = 0.6  # Example location for S2

    # Display ECG wave with S1 and S2 markers
    plt.figure(figsize=(10, 6))
    time = np.arange(0, len(audio)) / sr
    plt.plot(time, audio, color='b', label='ECG Waveform')
    plt.axvline(x=s1_location, color='r', linestyle='--', label='S1')
    plt.axvline(x=s2_location, color='g', linestyle='--', label='S2')
    plt.title('ECG Waveform with S1 and S2 Markers')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# GUI Setup
root = tk.Tk()
root.title("ECG Visualization")

record_button = tk.Button(root, text="Record Audio", command=record_audio)
record_button.pack()

plot_wave_button = tk.Button(root, text="Plot ECG Wave", command=plot_ecg_wave)
plot_wave_button.pack()

root.mainloop()
