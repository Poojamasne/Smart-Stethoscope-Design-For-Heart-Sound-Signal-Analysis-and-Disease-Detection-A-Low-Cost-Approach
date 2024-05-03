
from cProfile import label
import librosa
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif

# Load a wave file
audio_data, sr = librosa.load('D1.wav')

# Extract MFCC features
mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=20)  # Extract 20 MFCC coefficients

# Transpose the matrix to have feature vectors as rows instead of columns
mfccs = np.transpose(mfccs)

# Extract features using SelectKBest (F-test)
selected_features = SelectKBest(f_classif, k=10).fit_transform(mfccs,label)  # Adjust k as needed

print("Selected features shape:", selected_features.shape)

