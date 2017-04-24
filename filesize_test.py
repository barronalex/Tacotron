from __future__ import print_function

import librosa
import numpy as np
from tqdm import tqdm

def griffinlim(spectrogram, n_iter = 100, window = 'hann', n_fft = 2048, win_length = 2048, hop_length = -1, verbose = False):
    if hop_length == -1:
        hop_length = n_fft // 4

    angles = np.exp(2j * np.pi * np.random.rand(*spectrogram.shape))

    t = tqdm(range(n_iter), ncols=100, mininterval=2.0, disable=not verbose)
    for i in t:
        full = np.abs(spectrogram).astype(np.complex) * angles
        inverse = librosa.istft(full, hop_length = hop_length, win_length = win_length, window = window)
        rebuilt = librosa.stft(inverse, n_fft = n_fft, hop_length = hop_length, win_length = win_length, window = window)
        angles = np.exp(1j * np.angle(rebuilt))

        if verbose:
            diff = np.abs(spectrogram) - np.abs(rebuilt)
            t.set_postfix(loss=np.linalg.norm(diff, 'fro'))

    full = np.abs(spectrogram).astype(np.complex) * angles
    inverse = librosa.istft(full, hop_length = hop_length, win_length = win_length, window = window)

    return inverse

fname = 'data/VCTK-Corpus/wav48/p225/p225_004.wav'
wave, sr = librosa.load(fname, sr=48000)
print(wave.shape)
print(sr)
stft = librosa.stft(wave, win_length=1200, hop_length=300)
print(stft.shape)
#mel = librosa.feature.melspectrogram(y=wave, n_mels=80)
#print(wave.shape)
#print(mel.shape)

stft = np.abs(stft)

inv = griffinlim(stft, hop_length=300, win_length=1200)
print(inv.shape)
librosa.output.write_wav('test_inv.wav', inv, sr)


#print(wave)
#inverted = librosa.istft(mag, win_length=1200, hop_length=300)
#with open('test.npy', 'wb') as f:
    #np.save(f, mel, allow_pickle=False)

#audio = np.array(wave, dtype=np.float64)
#print(audio.shape)

#fs, audio = wavfile.read(fname)

#audio = np.array(audio, dtype=np.float64)
#print(audio.shape)

#spectogram = stft(audio, fftsize=2048, step=700, compute_onesided=False)
#print(spectogram.dtype)
#spectogram = np.log(np.abs(spectogram))
#print(spectogram.dtype)
#print(spectogram.shape)
#print(np.prod(spectogram.shape))
#inv = iterate_invert_spectrogram(np.exp(spectogram), 2048, 700)


#wavfile.write('test_inv.wav', fs, soundsc(inv))










