import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from PIL import Image, ImageTk
import sounddevice as sd
import soundfile as sf
import librosa
import librosa.display
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Initialize model_accuracy variable
model_accuracy = None

# Function to record audio
def record_audio():
    duration = 5  # Recording duration in seconds
    sample_rate = 44100  # Sampling rate
    channels = 1  # Mono audio

    print("Recording... Press Ctrl+C to stop.")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, dtype='float32')
    sd.wait()
    sf.write('recorded_heart_sound.wav', audio, sample_rate)
    print("Recording saved as 'recorded_heart_sound.wav'")

# Function to remove noise from recorded audio
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

# Function to extract features from recorded audio
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

# Function to train the classification model
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

# Function to classify the patient based on the trained model
def execute_classify():
    global model_accuracy  # Access the global variable
    if model_accuracy is not None:
        # Check a threshold for classification (assuming >= 0.5 means patient has a disease)
        if model_accuracy >= 0.5:
            result_label.config(text="Patient has a Disease", fg="red")
            # Provide advice to consult a doctor if the patient has a disease
            advice_label.config(text="Please consult with a doctor", fg="red")
        else:
            result_label.config(text="Patient is Normal", fg="green")
            # Clear the advice label if the patient is normal
            advice_label.config(text="", fg="green")
    else:
        messagebox.showinfo("Error", "Model accuracy is not available. Train the model first.")

# Initialize the Tkinter GUI
root = tk.Tk()
root.title("Smart Stethoscope")

# Define labels
accuracy_label = ttk.Label(root, text="Model Accuracy: ")
accuracy_label.pack()

result_label = ttk.Label(root, text="")
result_label.pack()

advice_label = ttk.Label(root, text="")
advice_label.pack()

# Add a custom icon
icon = Image.open("icon.png")
icon = icon.resize((64, 64), Image.LANCZOS)  # Resampling method: LANCZOS
icon = ImageTk.PhotoImage(icon)
root.iconphoto(False, icon)

# Create a frame for the buttons
button_frame = ttk.Frame(root)
button_frame.pack(pady=10)

# Create buttons and labels
record_button = ttk.Button(button_frame, text="Record Audio", command=record_audio)
record_button.grid(row=0, column=0, padx=5)

remove_noise_button = ttk.Button(button_frame, text="Remove Noise", command=remove_noise)
remove_noise_button.grid(row=0, column=1, padx=5)

extract_features_button = ttk.Button(button_frame, text="Extract Features", command=extract_features)
extract_features_button.grid(row=0, column=2, padx=5)

train_button = ttk.Button(button_frame, text="Train Model", command=train_model)
train_button.grid(row=0, column=3, padx=5)

classify_button = ttk.Button(button_frame, text="Classify", command=execute_classify)
classify_button.grid(row=0, column=4, padx=5)
