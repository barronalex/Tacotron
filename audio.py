from __future__ import print_function
from __future__ import division

import librosa
import numpy as np
from tqdm import tqdm

n_fft = 2048
win_length = 1200
hop_length = int(win_length/4)

# NOTE: If you change the decoder output width r, make sure to rerun preprocess.py.
# it stores the arrays in a different format based on this value
r = 5

# To allow the decoder to output multiple non-overlapping frames at each time step
# we need to reshape the frames so that they appear in non-overlapping chunks
# of length r
# we then reshape these frames back to the normal overlapping representation to be outputted
def reshape_frames(signal, forward=True):
    if forward:
        pad_length = signal.shape[1] % (4*r)
        pad_length = 4*r - pad_length if pad_length > 0 else 0
        signal = np.pad(signal, ((0,0), (0,pad_length)), 'constant', constant_values=0)

        split_points = np.arange(4*r, signal.shape[1]+1, step=4*r)
        splits = np.split(signal, split_points, axis=1)
        new_signal = np.concatenate([np.concatenate(np.split(s, r, axis=1), axis=0) for s in splits[:-1]], axis=1)
        return new_signal.T
    else:
        signal = np.reshape(signal, (-1, int(signal.shape[1]/r)))
        split_points = np.arange(4*r, signal.shape[0]+1, step=4*r)
        splits = np.split(signal, split_points, axis=0)
        new_signal = np.concatenate([np.concatenate(np.split(s, s.shape[0]/r, axis=0), axis=1) for s in splits[:-1]], axis=0)
        new_signal = np.reshape(new_signal, (-1, signal.shape[1]))
        return new_signal
        

def process_wav(fname, n_fft=2048, win_length=1200, hop_length=300, sr=16000):
    wave, sr = librosa.load(fname, mono=True, sr=sr)

    pre_emphasis = 0.97
    wave = np.append(wave[0], wave[1:] - pre_emphasis * wave[:-1])

    stft = librosa.stft(wave, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
    mel = librosa.feature.melspectrogram(S=stft, n_mels=80)

    stft = np.log(np.abs(stft))
    mel = np.log(np.abs(mel))

    stft = reshape_frames(stft)
    mel = reshape_frames(mel)

    return mel, stft

def invert_spectrogram(spec, out_fn=None, sr=16000):
    spec = reshape_frames(spec, forward=False)

    inv = griffinlim(np.exp(spec.T), n_fft=n_fft, win_length=win_length, hop_length=hop_length, verbose=False)
    if out_fn is not None:
        librosa.output.write_wav(out_fn, inv, sr)
    return inv

# lightly adapted from https://github.com/librosa/librosa/issues/434
def griffinlim(spectrogram, n_iter = 50, window = 'hann', n_fft = 2048, win_length = 2048, hop_length = -1, verbose = False):
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

if __name__ == '__main__':
    # simple tests
    fname = 'data/VCTK-Corpus/wav48/p225/p225_004.wav'
    mel, stft = process_wav(fname)

    print(stft.shape)
    mean = np.mean(stft, axis=0)
    print(mean.shape)

    print(mean)
    news = stft - mean

    print(np.sum(np.abs(news - stft)))

    print(np.sum(mean))

    olds = news + mean

    print('might be zero')
    print(np.sum(np.abs(olds - stft)))
    print(np.sum(np.abs(news - olds)))

    invert_spectrogram(stft, 'test_inv.wav')

    test = np.repeat(np.arange(36)[:, None] + 1, 7, axis=1)
    out = reshape_frames(test.T)

    inv = reshape_frames(out, forward=False)

    assert np.array_equal(test, inv[:test.shape[0]])


