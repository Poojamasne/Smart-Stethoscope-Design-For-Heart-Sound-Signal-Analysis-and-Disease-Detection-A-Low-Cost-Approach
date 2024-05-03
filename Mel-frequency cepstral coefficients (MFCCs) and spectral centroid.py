import librosa
import numpy as np

# Load the heart sound WAV file
file_path = 'C:\\Users\\Pooja\\OneDrive\\Desktop\\power spetrum\\Normal heart sound\\D1.wav'

y, sr = librosa.load(file_path)

# Extract features using Librosa
# Example features: Mel-frequency cepstral coefficients (MFCCs) and spectral centroid
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)

# Calculate statistical features from MFCCs
mfccs_mean = np.mean(mfccs, axis=1)
mfccs_std = np.std(mfccs, axis=1)

# Print and visualize the features
print("MFCCs Mean:", mfccs_mean)
print("MFCCs Standard Deviation:", mfccs_std)
print("Spectral Centroids:", spectral_centroids)

# You can continue with various other features and machine learning techniques as needed.
