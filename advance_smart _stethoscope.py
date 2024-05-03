import tkinter as tk
import sounddevice as sd
import soundfile as sf
import librosa
import librosa.display
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from PIL import Image, ImageTk

def record_audio():
    duration = 5  # Recording duration in seconds
    sample_rate = 44100  # Sampling rate
    channels = 1  # Mono audio

    print("Recording... Press Ctrl+C to stop.")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, dtype='float32')
    sd.wait()
    sf.write('recorded_heart_sound.wav', audio, sample_rate)
    print("Recording saved as 'recorded_heart_sound.wav'")

def remove_noise():
    audio_file = 'recorded_heart_sound.wav'
    audio, sr = librosa.load(audio_file, sr=None)
    reduced_audio = librosa.effects.preemphasis(audio)
    
    plt.figure(figsize=(8, 6))
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(audio, sr=sr)
    plt.title('Original Audio Waveform')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    librosa.display.waveshow(reduced_audio, sr=sr)
    plt.title('Denoised Audio Waveform')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def extract_features():
    audio_file = 'recorded_heart_sound.wav'
    audio, sr = librosa.load(audio_file, sr=None)
    reduced_audio = librosa.effects.preemphasis(audio)

    # Calculate spectral centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=reduced_audio, sr=sr)[0]
    
    # Calculate spectral bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=reduced_audio, sr=sr)[0]
    
    # Calculate spectral contrast
    spectral_contrast = librosa.feature.spectral_contrast(y=reduced_audio, sr=sr)[0]
    
    # Calculate spectral rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(y=reduced_audio, sr=sr)[0]

    # Plotting the features (you can remove this if not needed)
    plt.figure(figsize=(10, 6))

    plt.subplot(4, 1, 1)
    librosa.display.waveshow(reduced_audio, sr=sr)
    plt.title('Denoised Audio Waveform')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.subplot(4, 1, 2)
    plt.plot(spectral_centroid, label='Spectral Centroid')
    plt.title('Spectral Centroid')
    plt.xlabel('Frame')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 1, 3)
    plt.plot(spectral_bandwidth, label='Spectral Bandwidth')
    plt.title('Spectral Bandwidth')
    plt.xlabel('Frame')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)

    plt.subplot(4, 1, 4)
    plt.plot(spectral_rolloff, label='Spectral Rolloff')
    plt.title('Spectral Rolloff')
    plt.xlabel('Frame')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Returning the extracted features
    return {
        'Spectral Centroid': spectral_centroid,
        'Spectral Bandwidth': spectral_bandwidth,
        'Spectral Contrast': spectral_contrast,
        'Spectral Rolloff': spectral_rolloff
    }


def train_model():
    global model_accuracy  # Access the global variable
    # Load and preprocess your dataset, extract features, and labels
    # Replace 'features' and 'labels' with your actual data
    features = np.random.rand(100, 20)  # Example feature matrix
    labels = np.random.randint(0, 2, size=100)  # Example labels (binary classification)
    
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    clf = SVC(kernel='linear')
    clf.fit(X_train, y_train)

    # Evaluate the trained model
    model_accuracy = clf.score(X_test, y_test)
    print(f"Model accuracy: {model_accuracy}")

    # Display the model accuracy on the GUI
    accuracy_label.config(text=f"Model Accuracy: {model_accuracy:.2%}")


def classify():
    # Implement classification logic here using the trained model
    pass
    #accuracy = 0 # Replace with the actual accuracy score
    #return accuracy

def plot_ecg_wave():
    audio_file = 'recorded_heart_sound.wav'
    audio, sr = librosa.load(audio_file, sr=None)
    
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

def execute_record():
    record_audio()

def execute_remove_noise():
    remove_noise()

def execute_extract_features():
    extract_features()

def execute_train():
    train_model()


def execute_classify():
    global model_accuracy  # Access the global variable
    if model_accuracy is not None:
        # Check a threshold for classification
    
        # Check a threshold for classification (assuming >= 0.5 means patient has a disease)
        if model_accuracy >= 1:
            result_label.config(text="Patient has a Disease")
            # Provide advice to consult a doctor if the patient has a disease
            advice_label.config(text="Please consult with a doctor")
        else:
            result_label.config(text="Patient is Normal")
            # Clear the advice label if the patient is normal
            advice_label.config(text="")
    else:
        print("Model accuracy is not available. Train the model first.")


# GUI Setup
root = tk.Tk()
root.title("Smart Stethoscope")

record_button = tk.Button(root, text="Record Audio", command=execute_record)
record_button.pack()

remove_noise_button = tk.Button(root, text="Remove Noise", command=execute_remove_noise)
remove_noise_button.pack()

#plot_wave_button = tk.Button(root, text="Plot ECG Wave", command=plot_ecg_wave)
#plot_wave_button.pack()

extract_features_button = tk.Button(root, text="Extract Features", command=execute_extract_features)
extract_features_button.pack()

train_button = tk.Button(root, text="Train Model", command=execute_train)
train_button.pack()

accuracy_label = tk.Label(root, text="Accuracy: ")
accuracy_label.pack()

classify_button = tk.Button(root, text="Disease Detection", command=execute_classify)
classify_button.pack()

plot_wave_button = tk.Button(root, text="Plot ECG Wave", command=plot_ecg_wave)
plot_wave_button.pack()

# Label to display advice
advice_label = tk.Label(root, text="")
advice_label.pack()

result_label = tk.Label(root, text="")
result_label.pack()

root.mainloop()
