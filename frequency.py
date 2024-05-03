import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram

# Step 1: Read heart sound data from a WAV file
sample_rate, audio_data = wavfile.read('D34.wav')

# Step 2: No need for signal processing if it's already mono

# Step 3: Compute Power Spectrum using Spectrogram
frequencies, times, power_spectrum = spectrogram(audio_data, fs=sample_rate)

# Step 4: Visualization
plt.figure(figsize=(12, 8))

# Plot Heart Sound Waveform
plt.subplot(2, 1, 1)
plt.plot(np.arange(len(audio_data)) / sample_rate, audio_data)
plt.title('Heart Sound Waveform')
plt.grid(True)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

# Plot Power Spectrum
plt.subplot(2, 1, 2)

# Calculate the power spectrum using FFT
frequencies = np.fft.fftfreq(len(audio_data), d=1/sample_rate)
fft_values = np.fft.fft(audio_data)
power_spectrum = np.abs(fft_values)**2

# Plot the power spectrum
#plt.figure(figsize=(10, 6))
plt.plot(frequencies, power_spectrum)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.title('Power Spectrum-Normal Heart Sound')
plt.grid(True)
plt.xlim(0, sample_rate/2)  # Display only positive frequencies
plt.show()