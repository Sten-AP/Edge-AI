from sys import byteorder
from array import array
from struct import pack
import time
import pyaudio
import wave
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

MAXIMUM = 16384
THRESHOLD = 2000
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
RATE = 16000
BASE_DIR = os.path.dirname(__file__)
DATASET_PATH = f'{BASE_DIR}/data'

model = keras.models.load_model(f"{BASE_DIR}/model")

def get_spectrogram(waveform):
    spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram[tf.newaxis,...]

def is_silent(snd_data):
    "Returns 'True' if below the 'silent' threshold"
    print(max(snd_data))
    return max(snd_data) < THRESHOLD

def normalize(snd_data):
    "Average the volume out"
    times = float(MAXIMUM)/max(abs(i) for i in snd_data)

    r = array('h')
    for i in snd_data:
        r.append(int(i*times))
    return r

def trim(snd_data):
    "Trim the blank spots at the start and end"
    def _trim(snd_data):
        snd_started = False
        r = array('h')

        for i in snd_data:
            if not snd_started and abs(i)>THRESHOLD:
                snd_started = True
                r.append(i)

            elif snd_started:
                r.append(i)
        return r

    # Trim to the left
    snd_data = _trim(snd_data)

    # Trim to the right
    snd_data.reverse()
    snd_data = _trim(snd_data)
    snd_data.reverse()
    return snd_data

def add_silence(snd_data, seconds):
    "Add silence to the start and end of 'snd_data' of length 'seconds' (float)"
    silence = [0] * int(seconds * RATE)
    r = array('h', silence)
    r.extend(snd_data)
    r.extend(silence)
    return r

def record():
    """
    Record a word or words from the microphone and 
    return the data as an array of signed shorts.

    Normalizes the audio, trims silence from the 
    start and end, and pads with 0.5 seconds of 
    blank sound to make sure VLC et al can play 
    it without getting chopped off.
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=RATE,
        input=True, output=True,
        frames_per_buffer=CHUNK_SIZE)

    num_silent = 0
    snd_started = False

    r = array('h')

    while 1:
        # little endian, signed short
        snd_data = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            snd_data.byteswap()
        r.extend(snd_data)

        silent = is_silent(snd_data)

        if silent and snd_started:
            num_silent += 1
        elif not silent and not snd_started:
            snd_started = True

        if snd_started and num_silent > 10:
            break

    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    r = normalize(r)
    r = trim(r)
    r = add_silence(r, 0)
    return sample_width, r

def record_to_file(path):
    "Records from the microphone and outputs the resulting data to 'path'"
    sample_width, data = record()
    data = pack('<' + ('h'*len(data)), *data)

    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()

if __name__ == '__main__':
    while True:
        time.sleep(1)
        print("please speak a word into the microphone")
        record_to_file(f'{DATASET_PATH}/audio_input.wav')
        print(f"done - result written to audio_input.wav")
        
        input_data = f'{DATASET_PATH}/audio_input.wav'
        input_data = tf.io.read_file(str(input_data))
        input_data, sample_rate = tf.audio.decode_wav(input_data, desired_channels=1, desired_samples=16000)
        input_data = tf.squeeze(input_data, axis=-1)
        waveform = get_spectrogram(input_data)

        prediction = model.predict(waveform)
        x_labels = ['aardbei', 'boom', 'disco', 'gras', 'kaas', 'kers', 'zon']
        index = (np.argmax(prediction[0]))
        print(x_labels[index])
        
        # plt.bar(x_labels, tf.nn.softmax(prediction[0]))
        # plt.title(x_labels[index])
        # plt.show()
