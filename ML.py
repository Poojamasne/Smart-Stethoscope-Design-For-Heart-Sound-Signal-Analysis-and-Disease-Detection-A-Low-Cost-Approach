import os
import librosa
import numpy as np
from sklearn.svm import SVC

# Function to extract power spectrum features from audio files
def extract_features(file_path):
    audio_data, _ = librosa.load(file_path, sr=None)  # Load audio file
    power_spectrum = np.abs(librosa.stft(audio_data))**2  # Compute power spectrum
    mean_spectrum = np.mean(power_spectrum, axis=1)  # Calculate mean power across time
    return mean_spectrum

import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Set absolute paths to the 'normal' and 'abnormal' folders containing .wav files
normal_folder = r'C:\Users\Pooja\OneDrive\Desktop\power spetrum\normal'
abnormal_folder = r'C:\Users\Pooja\OneDrive\Desktop\power spetrum\abnormal'

# Collect file paths for normal and abnormal audio files
normal_files = [os.path.join(normal_folder, file) for file in os.listdir(normal_folder) if file.endswith('.wav')]
abnormal_files = [os.path.join(abnormal_folder, file) for file in os.listdir(abnormal_folder) if file.endswith('.wav')]

# ... (rest of the code remains unchanged)


# Create labels for normal (0) and abnormal (1) classes
normal_labels = [0] * len(normal_files)
abnormal_labels = [1] * len(abnormal_files)

# Combine the file paths and labels
file_paths = normal_files + abnormal_files
labels = normal_labels + abnormal_labels

# Extract features for each audio file
features = [extract_features(file) for file in file_paths]

# Initialize and train an SVM classifier
clf = SVC(kernel='linear', random_state=42)
clf.fit(features, labels)

# Now, let's predict the label for a new unseen file
unseen_file_path = 'C:\\Users\\Pooja\\OneDrive\\Desktop\\power spetrum\\pythoncode\\denoised_heart_sound.wav'
  # Replace with the path of the new/unseen audio file

# Extract features for the unseen file
unseen_features = extract_features(unseen_file_path)
unseen_features = np.array(unseen_features).reshape(1, -1)  # Reshape for single sample prediction

# Make prediction using the trained model
prediction = clf.predict(unseen_features)

# Display the predicted label
if prediction[0] == 0:
    print("The patient is classified as 'normal'.")
else:
    print("The patient is classified as 'abnormal/disease'.")
