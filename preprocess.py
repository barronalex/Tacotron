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

DATA_DIR = 'data/'

vocab = {}
ivocab = {}

def process_char(char):
    if not char in vocab:
        next_index = len(vocab)
        vocab[char] = next_index
        ivocab[next_index] = char
    return vocab[char]

def make_sequence_example(stft, mel, text, speaker):
    assert stft.shape[0] == 1025
    stft = stft.flatten()
    mel = mel.flatten()

    sequence = tf.train.SequenceExample()

    sequence.context.feature['speech_length'].int64_list.value.append(len(stft))
    sequence.context.feature['text_length'].int64_list.value.append(len(text))
    sequence.context.feature['speaker'].int64_list.value.append(int(speaker))

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


def preprocess_vctk():
    # adapted from https://github.com/buriburisuri/speech-to-text-wavenet/blob/master/preprocess.py

    # set up TensorFlow proto
    with open(DATA_DIR + 'VCTK-Corpus/train.proto', 'w') as pf:
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
            wave_file = DATA_DIR + 'VCTK-Corpus/wav48/%s/' % f[:4] + f + '.wav'
            txt_file = DATA_DIR + 'VCTK-Corpus/txt/%s/' % f[:4] + f + '.txt'

            wave, sr = librosa.load(wave_file, mono=True, sr=24000)

            stft = librosa.stft(wave, n_fft=2048, win_length=1200, hop_length=300)
            stft = np.log(np.abs(stft))

            mel = librosa.feature.melspectrogram(S=stft, n_mels=80)
            mel = np.log(np.abs(mel))
            assert stft.shape[1] == mel.shape[1]

            with open(txt_file, 'r') as tff:
                text = tff.read().strip()
                text = [process_char(c) for c in list(text)]
                #TODO possibly normalize text here?
            
            speaker = f[1:4]
            if speaker != '225': break

            sequence = make_sequence_example(stft, mel, text, speaker)
            writer.write(sequence.SerializeToString())
        writer.close()

        # save vocabulary
        with open(DATA_DIR + 'vocab.pkl', 'wb') as vf:
            pkl.dump(ivocab, vf)
        

if __name__ == '__main__':
    preprocess_vctk()



