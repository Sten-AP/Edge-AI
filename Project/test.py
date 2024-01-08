import os
import wave
from array import array


RATE = 44100
BASE_DIR = str(os.getcwdb())[2:-1]
DATASET_PATH = f'{BASE_DIR}/Project/data'
LABELS = ['aardbei', 'boom', 'gras', 'kaas', 'kers', 'zon']

input_data = f'{DATASET_PATH}/audio_input.wav'
# input_data = tf.io.read_file(str(input_data))
# input_data, sample_rate = tf.audio.decode_wav(input_data, desired_channels=1, desired_samples=RATE)
# pprint(sample_rate)


def add_silence(snd_data):
    "Add silence to the start and end of 'snd_data' of length 'seconds' (float)"
    silence = [0] * int((RATE - snd_data.getnframes())/2)
    r = array('h', silence)
    r.extend(snd_data)
    r.extend(silence)
    while len(r) < RATE:
        r.extend([0])
    return r