import librosa
import numpy as np
from scipy.stats import skew, kurtosis

def extract_features(audio_file):
    # Load the heart sound audio file
    y, sr = librosa.load(audio_file)

    # Extract features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
    rms_energy = librosa.feature.rms(y=y)[0]
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

    return mfccs, chroma, spectral_centroid, spectral_bandwidth, zero_crossing_rate, rms_energy, spectral_contrast

def calculate_statistics(features, feature_names):
    for feature, name in zip(features, feature_names):
        print(f"\n{name} Feature Statistics:")
        print(f"Mean: {np.mean(feature, axis=1)}" if feature.ndim > 1 else f"Mean: {np.mean(feature)}")
        print(f"Standard Deviation: {np.std(feature, axis=1)}" if feature.ndim > 1 else f"Standard Deviation: {np.std(feature)}")
        print(f"Skewness: {skew(feature, axis=1)}" if feature.ndim > 1 else f"Skewness: {skew(feature)}")
        print(f"Kurtosis: {kurtosis(feature, axis=1)}" if feature.ndim > 1 else f"Kurtosis: {kurtosis(feature)}")

if __name__ == "__main__":
    # Replace 'path/to/heart_sound.wav' with the actual path to your heart sound audio file
    audio_file_path = 'D1.wav'

    # Extract features
    extracted_features = extract_features(audio_file_path)

    # Define feature names
    feature_names = ['MFCCs', 'Chroma', 'Spectral Centroid', 'Spectral Bandwidth', 'Zero Crossing Rate', 'RMS Energy', 'Spectral Contrast']

    # Calculate statistical measures for the extracted features
    calculate_statistics(extracted_features, feature_names)
