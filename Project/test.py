import tensorflow as tf
import os
from pprint import pprint
import wave
from array import array




MAXIMUM = 16384
THRESHOLD = 3000
CHUNK_SIZE = 1024
RATE = 44100
BASE_DIR = str(os.getcwdb())[2:-1]
DATASET_PATH = f'{BASE_DIR}/data'
LABELS = ['aardbei', 'boom', 'gras', 'kaas', 'kers', 'zon']

input_data = f'{DATASET_PATH}/audio_input.wav'
# input_data = tf.io.read_file(str(input_data))
# input_data, sample_rate = tf.audio.decode_wav(input_data, desired_channels=1, desired_samples=RATE)
# pprint(sample_rate)

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


w = wave.open(input_data, 'r')
index = 0
for i in range(w.getnframes()):
    frame = w.readframes(i)
    index+=1
print(index/RATE)
