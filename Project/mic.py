from gpio import toggle_led
from pyaudio import PyAudio, paInt16
from sys import byteorder
from array import array
from struct import pack
from time import sleep
from wave import open
import os
import tensorflow as tf
from numpy import random, argmax
import matplotlib.pyplot as plt
import tflite_runtime.interpreter as tflite
import threading

seed = 42
tf.random.set_seed(seed)
random.seed(seed)

MAXIMUM = 16384
THRESHOLD = 2000
CHUNK_SIZE = 1024
FORMAT = paInt16
RATE = 48000
BASE_DIR = str(os.getcwdb())[2:-1]
DATASET_PATH = f'{BASE_DIR}/data'
LABELS = ['aardbei', 'boom', 'disco', 'gras', 'kaas', 'kers', 'zon']

interpreter = tflite.Interpreter(model_path=f"{BASE_DIR}/model.tflite")
interpreter.allocate_tensors()

def get_spectrogram(waveform):
    spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram[tf.newaxis,...]

def is_silent(snd_data):
    "Returns 'True' if below the 'silent' threshold"
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
                print(abs(i))
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
    p = PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=RATE,
        input=True, output=True,
        frames_per_buffer=CHUNK_SIZE, input_device_index=0)

    num_silent = 0
    snd_started = False

    r = array('h')

    while True:
        snd_data = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            snd_data.byteswap()
        r.extend(snd_data)

        silent = is_silent(snd_data)

        if silent and snd_started:
            num_silent += 1
        elif not silent and not snd_started:
            snd_started = True

        if snd_started and num_silent > 5:
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

    wf = open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()

def main():
    input_data = f'{DATASET_PATH}/audio_input.wav'
    input_data = tf.io.read_file(str(input_data))
    input_data, sample_rate = tf.audio.decode_wav(input_data, desired_channels=1, desired_samples=48000)
    input_data = tf.squeeze(input_data, axis=-1)
    waveform = get_spectrogram(input_data)
        
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]["index"], waveform)
    interpreter.invoke()

    output_details = interpreter.get_output_details()
    output_data = interpreter.get_tensor(output_details[0]["index"])
        
    index = argmax(output_data[0])
    prediction = LABELS[index]
    confidense = int(tf.nn.softmax(output_data[0])[index].numpy() * 100)
        
    print(f"Prediction: {prediction} - Confidence: {confidense}%")
    # plt.bar(LABELS, tf.nn.softmax(output_data[0]))
    # plt.title(prediction)
    # plt.show()
        
    if confidense > 50:
        print(f"Confident enough about {prediction}\n")
        toggle_led(LABELS[index])
    else:
        print(f"Not confident about {prediction}...\n")

if __name__ == '__main__':
    while True:
        sleep(1)
        print("please speak a word into the microphone")
        record_to_file(f'{DATASET_PATH}/audio_input.wav')
        print(f"done - result written to audio_input.wav\n")
        
        predictThread = threading.Thread(target=main)
        predictThread.start()
            
