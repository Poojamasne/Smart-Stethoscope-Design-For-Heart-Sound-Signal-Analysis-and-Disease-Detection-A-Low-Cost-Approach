import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Function to extract features from audio files
def extract_features(file_path):
    try:
        audio_data, _ = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio_data, sr=22050, n_mfcc=40)
        mfccs_processed = np.mean(mfccs.T, axis=0)
        
    except Exception as e:
        print(f"Error encountered while parsing file: {file_path}")
        return None
    
    return mfccs_processed

# Path to the folders containing normal and abnormal audio files
normal_folder = 'C:\\Users\\Pooja\\OneDrive\\Desktop\\power spetrum\\Normal heart sound'
abnormal_folder = 'C:\\Users\\Pooja\\OneDrive\\Desktop\\power spetrum\\Abnormal heart sound'

# Collect features and labels
features = []
labels = []

# Extract features for normal audio files
for filename in os.listdir(normal_folder):
    file_path = os.path.join(normal_folder, filename)
    feature = extract_features(file_path)
    if feature is not None:
        features.append(feature)
        labels.append(0)  # Label 0 for normal

# Extract features for abnormal audio files
for filename in os.listdir(abnormal_folder):
    file_path = os.path.join(abnormal_folder, filename)
    feature = extract_features(file_path)
    if feature is not None:
        features.append(feature)
        labels.append(1)  # Label 1 for abnormal

# Convert lists to numpy arrays
X = np.array(features)
y = np.array(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the SVM classifier
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Predict on the test set
predictions = svm_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2f}")

# Display classification report
print("Classification Report:")
print(classification_report(y_test, predictions))
