import pyaudio
import wave
import os
import datetime

# ...

now = datetime.datetime.now()
filename = 'Wave_' + str(now)[:10] + now.strftime("_%H_%M_%S.wav")

form_1 = pyaudio.paInt16
chans = 1  # Change this to 1 or 2 based on your device's capabilities
samp_rate = 44100
chunk = 512
record_secs = 15     # record time
dev_index = 2
wav_output_filename = filename

audio = pyaudio.PyAudio()

# setup audio input stream
stream = audio.open(
    format=form_1,
    rate=samp_rate,
    channels=chans,
    input_device_index=dev_index,
    input=True,
    frames_per_buffer=chunk
)

print("recording")
frames = []

for _ in range(0, int((samp_rate / chunk) * record_secs)):
    data = stream.read(chunk, exception_on_overflow=False)
    frames.append(data)

print("finished recording")

stream.stop_stream()
stream.close()
audio.terminate()

# creates wave file with audio read in
wavefile = wave.open(wav_output_filename, 'wb')
wavefile.setnchannels(chans)
wavefile.setsampwidth(audio.get_sample_size(form_1))
wavefile.setframerate(samp_rate)
wavefile.writeframes(b''.join(frames))
wavefile.close()

# plays the audio file
os.system(f"aplay {wav_output_filename}")
