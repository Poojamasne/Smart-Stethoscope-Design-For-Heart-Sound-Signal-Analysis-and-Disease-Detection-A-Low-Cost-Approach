import sounddevice as sd
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np

# Set the duration of the recording in seconds
duration = 5
output_file = 'recorded_audio.wav'



# Record audio
print("Recording...")
audio_data = sd.rec(int(duration * 44100), samplerate=44100, channels=1, dtype='float64')
sd.wait()

# Flatten the audio data to a 1D array
audio_data = np.squeeze(audio_data)

# Create a time axis for the waveform plot
time = np.linspace(0, duration, len(audio_data))

# Plot the waveform
plt.figure(figsize=(8, 4))
plt.plot(time, audio_data, color='blue')
plt.title('Audio Waveform')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()
