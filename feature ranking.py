import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Function for feature extraction
def extract_features(file_path):
    y, sr = librosa.load(file_path)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    # Normalize the feature matrix to a fixed size (e.g., using mean or padding)
    normalized_features = np.hstack([np.mean(mfccs, axis=1), np.std(mfccs, axis=1)])
    return normalized_features

# Function to extract features from multiple WAV files
def extract_features_from_files(file_paths):
    all_features = []
    for file_path in file_paths:
        features = extract_features(file_path)
        all_features.append(features)
    return np.vstack(all_features)

# List of paths to your WAV files
# List of paths to your WAV files
wav_file_paths = [r'C:\Users\Pooja\OneDrive\Desktop\power spetrum\Normal heart sound\D1.wav',
                  r'C:\Users\Pooja\OneDrive\Desktop\power spetrum\Normal heart sound\D2.wav',
                  r'C:\Users\Pooja\OneDrive\Desktop\power spetrum\Normal heart sound\D3.wav']


# Extract features from multiple files
all_file_features = extract_features_from_files(wav_file_paths)

# Assuming you have labels for your data
labels = [0, 1, 0]  # Replace with your actual labels (should match the number of files)

# Combine features and labels
X = all_file_features
y = labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Get feature importances
feature_importances = clf.feature_importances_

# Sort the feature importances in descending order
indices = np.argsort(feature_importances)[::-1]

# Print feature ranking
print("Feature ranking:")
for f in range(X.shape[1]):
    print(f"{f + 1}. Feature {indices[f]} - importance: {feature_importances[indices[f]]}")
