import os
import wave
from array import array
import tensorflow as tf
from pprint import pprint
import numpy as np
RATE = 44100
BASE_DIR = str(os.getcwdb())[2:-1]
DATASET_PATH = f'{BASE_DIR}/Project/data'
LABELS = ['aardbei', 'boom', 'gras', 'kaas', 'kers', 'zon']

# input_data = f'{DATASET_PATH}/audio_input.wav'
# input_data = tf.io.read_file(str(input_data))
# input_data, sample_rate = tf.audio.decode_wav(input_data, desired_channels=1, desired_samples=RATE)
# pprint(sample_rate)


input_data = f'{DATASET_PATH}/audio_input.wav'
input_data = tf.io.read_file(str(input_data))
input_data, sample_rate = tf.audio.decode_wav(input_data, desired_channels=1)

fill_data = tf.zeros((int((RATE - input_data.shape[0]) / 2), 1))
input_data = tf.concat([fill_data, input_data, fill_data], axis=0)

while input_data.shape[0] < RATE:
    input_data = tf.concat([input_data, tf.zeros((1, 1))], axis=0)

input_data = tf.squeeze(input_data, axis=-1)
