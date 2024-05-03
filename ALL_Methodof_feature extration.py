import numpy as np
from scipy.signal import find_peaks, welch
import pywt

# Function to extract time-domain features
def time_domain_features(signal):
    envelope = np.abs(signal)  # Envelope of the signal
    energy = np.sum(np.square(signal))  # Energy of the signal
    zero_crossing_rate = np.sum(np.diff(np.sign(signal)) != 0) / len(signal)  # Zero-crossing rate
    mean_value = np.mean(signal)  # Mean value
    variance = np.var(signal)  # Variance
    skewness = np.mean((signal - np.mean(signal))**3) / np.std(signal)**3  # Skewness
    kurtosis = np.mean((signal - np.mean(signal))**4) / np.var(signal)**2  # Kurtosis
    return envelope, energy, zero_crossing_rate, mean_value, variance, skewness, kurtosis

# Function to extract frequency-domain features
def frequency_domain_features(signal, sampling_rate):
    f, Pxx = welch(signal, fs=sampling_rate)  # Power spectral density estimation
    spectral_centroid = np.sum(f * Pxx) / np.sum(Pxx)  # Spectral centroid
    spectral_bandwidth = np.sqrt(np.sum((f - spectral_centroid)**2 * Pxx) / np.sum(Pxx))  # Spectral bandwidth
    spectral_entropy = -np.sum(Pxx * np.log2(Pxx))  # Spectral entropy
    spectral_flatness = np.exp(np.mean(np.log(Pxx)))  # Spectral flatness
    return spectral_centroid, spectral_bandwidth, spectral_entropy, spectral_flatness

# Function to extract wavelet transform features
def wavelet_features(signal):
    coefficients = pywt.wavedec(signal, 'db4')  # Wavelet decomposition using Daubechies 4 wavelet
    features = []
    for coeff in coefficients:
        features.extend([np.mean(coeff), np.std(coeff), np.max(coeff), np.min(coeff)])
    return features

# Example usage
# Replace 'heart_sound_data' with your actual heart sound data
heart_sound_data = np.random.randn(1000)  # Simulated heart sound data

# Extract time-domain features
envelope, energy, zero_crossing_rate, mean_value, variance, skewness, kurtosis = time_domain_features(heart_sound_data)

# Extract frequency-domain features
sampling_rate = 44100  # Replace with your actual sampling rate
spectral_centroid, spectral_bandwidth, spectral_entropy, spectral_flatness = frequency_domain_features(heart_sound_data, sampling_rate)

# Extract wavelet transform features
wavelet_features = wavelet_features(heart_sound_data)

# Display or use the extracted features as needed
print("Time-domain features:")
print(f"Envelope: {envelope}")
print(f"Energy: {energy}")
print(f"Zero-crossing rate: {zero_crossing_rate}")
print(f"Mean value: {mean_value}")
print(f"Variance: {variance}")
print(f"Skewness: {skewness}")
print(f"Kurtosis: {kurtosis}")

print("\nFrequency-domain features:")
print(f"Spectral centroid: {spectral_centroid}")
print(f"Spectral bandwidth: {spectral_bandwidth}")
print(f"Spectral entropy: {spectral_entropy}")
print(f"Spectral flatness: {spectral_flatness}")

print("\nWavelet transform features:")
print(wavelet_features)
