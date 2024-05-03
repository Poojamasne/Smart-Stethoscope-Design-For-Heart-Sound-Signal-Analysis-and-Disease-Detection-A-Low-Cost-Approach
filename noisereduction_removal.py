import librosa
import librosa.display
import matplotlib.pyplot as plt

# Load the audio file
file_path = 'D1.wav'
audio, sr = librosa.load(file_path, sr=None)

# Display the original audio waveform
plt.figure(figsize=(8, 4))
librosa.display.waveshow(audio, sr=sr)
plt.title('Original Audio Waveform')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()

# Apply noise reduction using spectral subtraction
reduced_audio = librosa.effects.preemphasis(audio)

# Display the denoised audio waveform
plt.figure(figsize=(8, 4))
librosa.display.waveshow(reduced_audio, sr=sr)
plt.title('Denoised Audio Waveform')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()

# Save the denoised audio to a new file
output_file = 'denoised_audio.wav'
librosa.output.write_wav(output_file, reduced_audio, sr)
print(f"Denoised audio saved as '{output_file}'")
