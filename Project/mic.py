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
import librosa
import librosa.display
import soundfile as sf

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

THRESHOLD = 500
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
RATE = 16000
BASE_DIR = os.path.dirname(__file__)
DATASET_PATH = f'{BASE_DIR}\\data'

model = keras.models.load_model(f"{BASE_DIR}/model")

def filter_audio(input_file, output_file, threshold):
    # Laden van het audiobestand
    y, sr = librosa.load(input_file)

    # Berekenen van de energie van het geluid
    energy = librosa.feature.rms(y=y)

    # Bepalen van de frames boven de drempelwaarde
    frames_above_threshold = np.where(energy > threshold)

    # Filteren van de frames boven de drempelwaarde
    y_filtered = np.zeros_like(y)
    for frame_idx in frames_above_threshold[1]:
        start_sample = frame_idx * librosa.samples_to_frames(1)
        end_sample = (frame_idx + 1) * librosa.samples_to_frames(1)
        y_filtered[start_sample:end_sample] = y[start_sample:end_sample]

    # Opslaan van het gefilterde geluid
    sf.write(output_file, y_filtered, RATE)
    
def get_spectrogram(waveform):
    # Convert the waveform to a spectrogram via a STFT.
    spectrogram = tf.signal.stft(
        waveform, frame_length=255, frame_step=128)
    # Obtain the magnitude of the STFT.
    spectrogram = tf.abs(spectrogram)
    # Add a `channels` dimension, so that the spectrogram can be used
    # as image-like input data with convolution layers (which expect
    # shape (`batch_size`, `height`, `width`, `channels`).
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram[tf.newaxis,...]

def is_silent(snd_data):
    "Returns 'True' if below the 'silent' threshold"
    return max(snd_data) < THRESHOLD

def normalize(snd_data):
    "Average the volume out"
    MAXIMUM = 16384
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
        
        x = f'{DATASET_PATH}/audio_input.wav'
        y = f'{DATASET_PATH}/audio_input_filtered.wav'
        # filter_audio(x, y, 0)
        y = f'{DATASET_PATH}/audio_input.wav'
        y = tf.io.read_file(str(y))
        y, sample_rate = tf.audio.decode_wav(y, desired_channels=1, desired_samples=16000,)
        y = tf.squeeze(y, axis=-1)
        waveform = get_spectrogram(y)

        prediction = model.predict(waveform)
        # x_labels = ['no', 'yes', 'down', 'go', 'left', 'up', 'right', 'stop']
        x_labels = ['aardbei', 'boom', 'disco', 'gras', 'kaas', 'kers', 'zon']
        index = (np.argmax(prediction[0]))
        print(x_labels[index])
        
        plt.bar(x_labels, tf.nn.softmax(prediction[0]))
        plt.title(x_labels[index])
        plt.show()