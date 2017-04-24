from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import librosa
import os
import glob
import pandas as pd
import pickle as pkl
from tqdm import tqdm
import sys


mini = False
r = 5
DATA_DIR = 'data/'

vocab = {}
ivocab = {}

def process_char(char):
    if not char in vocab:
        next_index = len(vocab)
        vocab[char] = next_index
        ivocab[next_index] = char
    return vocab[char]

def reshape_frames(signal):
    pad_length = signal.shape[1] % (4*r)
    pad_length = 4*r - pad_length if pad_length > 0 else 0
    signal = np.pad(signal, ((0,0), (0,pad_length)), 'constant', constant_values=0)

    # rearrange frames based on value of r
    # split into sections of shape (80, 4*r)
    split_points = np.arange(4*r, signal.shape[1]+1, step=4*r)
    splits = np.split(signal, split_points, axis=1)
    new_signal = np.concatenate([np.concatenate(np.split(s, r, axis=1), axis=0) for s in splits[:-1]], axis=1)
    return new_signal.T

def make_sequence_example(stft, mel, text, speaker):
    assert stft.shape[0] == 1025

    # pad time dimension out to nearest multiple of 4
    # this allows us to easily interleave sequences of non-overlapping frames later

    sequence = tf.train.SequenceExample()

    mel = reshape_frames(mel)
    stft = reshape_frames(stft)
    print(mel.shape)
    print(stft.shape)

    sequence.context.feature['speech_length'].int64_list.value.append(mel.shape[0])
    sequence.context.feature['text_length'].int64_list.value.append(len(text))
    sequence.context.feature['speaker'].int64_list.value.append(int(speaker))

    stft = stft.flatten()
    mel = mel.flatten()

    mel_feature = sequence.feature_lists.feature_list["mel"]
    stft_feature = sequence.feature_lists.feature_list["stft"]

    text_feature = sequence.feature_lists.feature_list["text"]

    for s in stft:
        stft_feature.feature.add().float_list.value.append(s)

    for m in mel:
        mel_feature.feature.add().float_list.value.append(m)

    for c in text:
        text_feature.feature.add().int64_list.value.append(c)

    return sequence

def process_wav(fname, sr=24000):
    wave, sr = librosa.load(fname, mono=True, sr=24000)

    stft = librosa.stft(wave, n_fft=2048, win_length=1200, hop_length=300)
    stft = librosa.logamplitude(stft)

    mel = librosa.feature.melspectrogram(S=stft, n_mels=80)
    mel = librosa.logamplitude(mel)
    assert stft.shape[1] == mel.shape[1]
    return mel, stft

def preprocess_arctic():
    proto_file = DATA_DIR + 'cmu_us_slt_arctic/train.proto'

    with open(proto_file, 'w') as pf:
        writer = tf.python_io.TFRecordWriter(pf.name)

        txt_file = DATA_DIR + 'cmu_us_slt_arctic/etc/arctic.data'
        with open(txt_file, 'r') as tff:
            for line in tqdm(tff, total=1100):
                spl = line.split()
                id = spl[1]
                text = ' '.join(spl[2:-1])
                text = text[1:-1]
                text = [process_char(c) for c in list(text)]

                wav_file = DATA_DIR + 'cmu_us_slt_arctic/wav/{}.wav'.format(id)

                mel, stft = process_wav(wav_file, sr=16000)
                sequence = make_sequence_example(stft, mel, text, 0)
                writer.write(sequence.SerializeToString())

        writer.close()

        # save vocabulary
        with open(DATA_DIR + 'meta.pkl', 'wb') as vf:
            pkl.dump({'vocab': ivocab, 'r': r}, vf)




def preprocess_vctk():
    # adapted from https://github.com/buriburisuri/speech-to-text-wavenet/blob/master/preprocess.py

    if mini:
        proto_file = DATA_DIR + 'VCTK-Corpus/mini_train.proto'
    else:
        proto_file = DATA_DIR + 'VCTK-Corpus/train.proto'

    # set up TensorFlow proto
    with open(proto_file, 'w') as pf:
        writer = tf.python_io.TFRecordWriter(pf.name)

        # read label-info
        df = pd.read_table(DATA_DIR + 'VCTK-Corpus/speaker-info.txt', usecols=['ID'],
                           index_col=False, delim_whitespace=True)
        # read file IDs
        file_ids = []
        for d in [DATA_DIR + 'VCTK-Corpus/txt/p%d/' % uid for uid in df.ID.values]:
            file_ids.extend([f[-12:-4] for f in sorted(glob.glob(d + '*.txt'))])

        for i, f in tqdm(enumerate(file_ids), total=len(file_ids)):

            # wave file name
            wav_file = DATA_DIR + 'VCTK-Corpus/wav48/%s/' % f[:4] + f + '.wav'
            txt_file = DATA_DIR + 'VCTK-Corpus/txt/%s/' % f[:4] + f + '.txt'

            mel, stft = process_wav(wav_file)

            with open(txt_file, 'r') as tff:
                text = tff.read().strip()
                text = [process_char(c) for c in list(text)]
                #TODO possibly normalize text here?
            
            speaker = f[1:4]
            if mini and i > 9: break
            if speaker != '225': break

            sequence = make_sequence_example(stft, mel, text, speaker)
            writer.write(sequence.SerializeToString())
        writer.close()


    if mini:
        # save vocabulary and meta data
        with open(DATA_DIR + 'mini_meta.pkl', 'wb') as vf:
            pkl.dump({'vocab': ivocab, 'r': r}, vf)
    else:
        # save vocabulary
        with open(DATA_DIR + 'meta.pkl', 'wb') as vf:
            pkl.dump({'vocab': ivocab, 'r': r}, vf)

        

if __name__ == '__main__':
    preprocess_arctic()
    #preprocess_vctk()




