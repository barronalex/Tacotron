from __future__ import print_function

import librosa
import numpy as np

fname = 'data/VCTK-Corpus/wav48/p225/p225_004.wav'
wave, sr = librosa.load(fname, mono=True, sr=None)
stft = librosa.stft(wave)
mel = librosa.feature.melspectrogram(y=wave, n_mels=80)
print(wave.shape)
print(mel.shape)

print(wave)
#with open('test.npy', 'wb') as f:
    #np.save(f, mel, allow_pickle=False)




