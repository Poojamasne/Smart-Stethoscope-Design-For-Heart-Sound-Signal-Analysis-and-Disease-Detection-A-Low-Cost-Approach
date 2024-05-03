import wave
import numpy as np
import sys
import matplotlib.pyplot as plt

# Replace 'your_audio_file.wav' with your audio file path
audio_file = wave.open('D1.wav', 'rb')

# Read audio data
frames = audio_file.readframes(-1)

# Convert audio data to a NumPy array
raw = np.frombuffer(frames, dtype='int16')

# Print the data (optional)
print(raw)

# Close the audio file
audio_file.close()

# Check the number of audio channels
if audio_file.getnchannels() == 2:
    print("The audio has 2 channels (stereo).")
else:
    print("The audio has 1 channel (mono).")

# Close the audio file
audio_file.close()

plt.title("Wavefrom of wave file")
plt.plot(raw,color="blue")
plt.ylabel("Amplitude")
plt.xlabel("")
plt.show()