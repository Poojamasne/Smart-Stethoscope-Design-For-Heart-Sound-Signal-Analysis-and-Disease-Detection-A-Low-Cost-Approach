import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Function to extract MFCC features from audio file
def extract_features(file_path, num_mfcc=13, n_fft=2048, hop_length=512):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
    return mfccs

# Define your dataset directory containing audio files
#dataset_dir = 'dataset_dir ='C:\\Users\\Pooja\\OneDrive\\Desktop\\power spetrum\\pythoncode'
dataset_dir = 'C:\\Users\\Pooja\\OneDrive\\Desktop\\power spetrum\\pythoncode'

# Collect audio file paths and corresponding labels
audio_files = []
labels = []

for root, dirs, files in os.walk(dataset_dir):
    for file in files:
        if file.endswith('Wave_File_2020-06-21_07_49_56.wav'):
            audio_files.append(os.path.join(root, file))
            # Assign labels based on file names or a predefined mapping
            if "Aortic_stenosis_early" in file:
                labels.append(0)
            elif "AS_late" in file:
                labels.append(1)
            elif "Pulmonic_stenosis" in file:
                labels.append(2)

# Extract features and prepare data
X = []
for audio_file in audio_files:
    features = extract_features(audio_file)
    X.append(features)

X = np.array(X)
y = np.array(labels)



# Assuming X contains audio features and y contains corresponding labels
# Example: Features extracted using librosa and associated labels

# Ensure X and y have consistent number of samples
if len(X) != len(y):
    print("Number of samples in X and y are not equal.")
    # Handle this issue by aligning the number of samples or fixing misalignment

    # For example, you might remove extra samples from X or y
    min_samples = min(len(X), len(y))
    X = X[:min_samples]
    y = y[:min_samples]
    print(f"Number of samples adjusted to {min_samples}")

# Check the shapes after alignment
print(f"Shape of X: {len(X)}")
print(f"Shape of y: {len(y)}")

# Proceed with splitting the data into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


# Train a Support Vector Machine (SVM) classifier
svm = SVC()
svm.fit(X_train.reshape(X_train.shape[0], -1), y_train)

# Predict on the test set
y_pred = svm.predict(X_test.reshape(X_test.shape[0], -1))

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
