# License: BSD 3-clause
# Authors: Kyle Kastner
# Harvest, Cheaptrick, D4C, WORLD routines based on MATLAB code from M. Morise
# http://ml.cs.yamanashi.ac.jp/world/english/
# MGC code based on r9y9 (Ryuichi Yamamoto) MelGeneralizedCepstrums.jl
# Pieces also adapted from SPTK
from __future__ import division
import numpy as np
import scipy as sp
from numpy.lib.stride_tricks import as_strided
import scipy.signal as sg
from scipy.interpolate import interp1d
import wave
from scipy.cluster.vq import vq
from scipy import linalg, fftpack
from numpy.testing import assert_almost_equal
from scipy.linalg import svd
from scipy.io import wavfile
from scipy.signal import firwin
import zipfile
import tarfile
import os
import copy
import multiprocessing
from multiprocessing import Pool
import functools
import time
try:
    import urllib.request as urllib  # for backwards compatibility
except ImportError:
    import urllib2 as urllib


def download(url, server_fname, local_fname=None, progress_update_percentage=5,
             bypass_certificate_check=False):
    """
    An internet download utility modified from
    http://stackoverflow.com/questions/22676/
    how-do-i-download-a-file-over-http-using-python/22776#22776
    """
    if bypass_certificate_check:
        import ssl
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        u = urllib.urlopen(url, context=ctx)
    else:
        u = urllib.urlopen(url)
    if local_fname is None:
        local_fname = server_fname
    full_path = local_fname
    meta = u.info()
    with open(full_path, 'wb') as f:
        try:
            file_size = int(meta.get("Content-Length"))
        except TypeError:
            print("WARNING: Cannot get file size, displaying bytes instead!")
            file_size = 100
        print("Downloading: %s Bytes: %s" % (server_fname, file_size))
        file_size_dl = 0
        block_sz = int(1E7)
        p = 0
        while True:
            buffer = u.read(block_sz)
            if not buffer:
                break
            file_size_dl += len(buffer)
            f.write(buffer)
            if (file_size_dl * 100. / file_size) > p:
                status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl *
                                               100. / file_size)
                print(status)
                p += progress_update_percentage


def fetch_sample_speech_tapestry():
    url = "https://www.dropbox.com/s/qte66a7haqspq2g/tapestry.wav?dl=1"
    wav_path = "tapestry.wav"
    if not os.path.exists(wav_path):
        download(url, wav_path)
    fs, d = wavfile.read(wav_path)
    d = d.astype('float32') / (2 ** 15)
    # file is stereo? - just choose one channel
    return fs, d


def fetch_sample_file(wav_path):
    if not os.path.exists(wav_path):
        raise ValueError("Unable to find file at path %s" % wav_path)
    fs, d = wavfile.read(wav_path)
    d = d.astype('float32') / (2 ** 15)
    # file is stereo - just choose one channel
    if len(d.shape) > 1:
        d = d[:, 0]
    return fs, d


def fetch_sample_music():
    url = "http://www.music.helsinki.fi/tmt/opetus/uusmedia/esim/"
    url += "a2002011001-e02-16kHz.wav"
    wav_path = "test.wav"
    if not os.path.exists(wav_path):
        download(url, wav_path)
    fs, d = wavfile.read(wav_path)
    d = d.astype('float32') / (2 ** 15)
    # file is stereo - just choose one channel
    d = d[:, 0]
    return fs, d


def fetch_sample_speech_fruit(n_samples=None):
    url = 'https://dl.dropboxusercontent.com/u/15378192/audio.tar.gz'
    wav_path = "audio.tar.gz"
    if not os.path.exists(wav_path):
        download(url, wav_path)
    tf = tarfile.open(wav_path)
    wav_names = [fname for fname in tf.getnames()
                 if ".wav" in fname.split(os.sep)[-1]]
    speech = []
    print("Loading speech files...")
    for wav_name in wav_names[:n_samples]:
        f = tf.extractfile(wav_name)
        fs, d = wavfile.read(f)
        d = d.astype('float32') / (2 ** 15)
        speech.append(d)
    return fs, speech


def fetch_sample_speech_eustace(n_samples=None):
    """
    http://www.cstr.ed.ac.uk/projects/eustace/download.html
    """
    # data
    url = "http://www.cstr.ed.ac.uk/projects/eustace/down/eustace_wav.zip"
    wav_path = "eustace_wav.zip"
    if not os.path.exists(wav_path):
        download(url, wav_path)

    # labels
    url = "http://www.cstr.ed.ac.uk/projects/eustace/down/eustace_labels.zip"
    labels_path = "eustace_labels.zip"
    if not os.path.exists(labels_path):
        download(url, labels_path)

    # Read wavfiles
    # 16 kHz wav
    zf = zipfile.ZipFile(wav_path, 'r')
    wav_names = [fname for fname in zf.namelist()
                 if ".wav" in fname.split(os.sep)[-1]]
    fs = 16000
    speech = []
    print("Loading speech files...")
    for wav_name in wav_names[:n_samples]:
        wav_str = zf.read(wav_name)
        d = np.frombuffer(wav_str, dtype=np.int16)
        d = d.astype('float32') / (2 ** 15)
        speech.append(d)

    zf = zipfile.ZipFile(labels_path, 'r')
    label_names = [fname for fname in zf.namelist()
                   if ".lab" in fname.split(os.sep)[-1]]
    labels = []
    print("Loading label files...")
    for label_name in label_names[:n_samples]:
        label_file_str = zf.read(label_name)
        labels.append(label_file_str)
    return fs, speech


def stft(X, fftsize=128, step="half", mean_normalize=True, real=False,
         compute_onesided=True):
    """
    Compute STFT for 1D real valued input X
    """
    if real:
        local_fft = fftpack.rfft
        cut = -1
    else:
        local_fft = fftpack.fft
        cut = None
    if compute_onesided:
        cut = fftsize // 2 + 1
    if mean_normalize:
        X -= X.mean()
    if step == "half":
        X = halfoverlap(X, fftsize)
    else:
        X = overlap(X, fftsize, step)
    size = fftsize
    win = 0.54 - .46 * np.cos(2 * np.pi * np.arange(size) / (size - 1))
    X = X * win[None]
    X = local_fft(X)[:, :cut]
    return X


def istft(X, fftsize=128, step="half", wsola=False, mean_normalize=True,
          real=False, compute_onesided=True):
    """
    Compute ISTFT for STFT transformed X
    """
    if real:
        local_ifft = fftpack.irfft
        X_pad = np.zeros((X.shape[0], X.shape[1] + 1)) + 0j
        X_pad[:, :-1] = X
        X = X_pad
    else:
        local_ifft = fftpack.ifft
    if compute_onesided:
        X_pad = np.zeros((X.shape[0], 2 * X.shape[1])) + 0j
        X_pad[:, :fftsize // 2 + 1] = X
        X_pad[:, fftsize // 2 + 1:] = 0
        X = X_pad
    X = local_ifft(X).astype("float64")
    if step == "half":
        X = invert_halfoverlap(X)
    else:
        X = overlap_add(X, step, wsola=wsola)
    if mean_normalize:
        X -= np.mean(X)
    return X


def mdct_slow(X, dctsize=128):
    M = dctsize
    N = 2 * dctsize
    N_0 = (M + 1) / 2
    X = halfoverlap(X, N)
    X = sine_window(X)
    n, k = np.meshgrid(np.arange(N), np.arange(M))
    # Use transpose due to "samples as rows" convention
    tf = np.cos(np.pi * (n + N_0) * (k + 0.5) / M).T
    return np.dot(X, tf)


def imdct_slow(X, dctsize=128):
    M = dctsize
    N = 2 * dctsize
    N_0 = (M + 1) / 2
    N_4 = N / 4
    n, k = np.meshgrid(np.arange(N), np.arange(M))
    # inverse *is not* transposed
    tf = np.cos(np.pi * (n + N_0) * (k + 0.5) / M)
    X_r = np.dot(X, tf) / N_4
    X_r = sine_window(X_r)
    X = invert_halfoverlap(X_r)
    return X


def nsgcwin(fmin, fmax, n_bins, fs, signal_len, gamma):
    """
    Nonstationary Gabor window calculation

    References
    ----------
    Velasco G. A., Holighaus N., Dorfler M., Grill T.
    Constructing an invertible constant-Q transform with nonstationary Gabor
    frames, Proceedings of the 14th International Conference on Digital
    Audio Effects (DAFx 11), Paris, France, 2011

    Holighaus N., Dorfler M., Velasco G. A. and Grill T.
    A framework for invertible, real-time constant-Q transforms, submitted.

    Original matlab code copyright follows:

    AUTHOR(s) : Monika Dorfler, Gino Angelo Velasco, Nicki Holighaus, 2010-2011

    COPYRIGHT : (c) NUHAG, Dept.Math., University of Vienna, AUSTRIA
    http://nuhag.eu/
    Permission is granted to modify and re-distribute this
    code in any manner as long as this notice is preserved.
    All standard disclaimers apply.
    """
    # use a hanning window
    # no fractional shifts
    fftres = fs / signal_len
    fmin = float(fmin)
    fmax = float(fmax)
    gamma = float(gamma)
    nyq = fs / 2.
    b = np.floor(n_bins * np.log2(fmax / fmin))
    fbas = fmin * 2 ** (np.arange(b + 1) / float(n_bins))
    Q = 2 ** (1. / n_bins) - 2 ** (-1. / n_bins)
    cqtbw = Q * fbas + gamma
    cqtbw = cqtbw.ravel()
    maxidx = np.where(fbas + cqtbw / 2. > nyq)[0]
    if len(maxidx) > 0:
        # replicate bug in MATLAB version...
        # or is it a feature
        if sum(maxidx) == 0:
            first = len(cqtbw) - 1
        else:
            first = maxidx[0]
        fbas = fbas[:first]
        cqtbw = cqtbw[:first]
    minidx = np.where(fbas - cqtbw / 2. < 0)[0]
    if len(minidx) > 0:
        fbas = fbas[minidx[-1]+1:]
        cqtbw = cqtbw[minidx[-1]+1:]

    fbas_len = len(fbas)
    fbas_new = np.zeros((2 * (len(fbas) + 1)))
    fbas_new[1:len(fbas) + 1] = fbas
    fbas = fbas_new
    fbas[fbas_len + 1] = nyq
    fbas[fbas_len + 2:] = fs - fbas[1:fbas_len + 1][::-1]
    bw = np.zeros_like(fbas)
    bw[0] = 2 * fmin
    bw[1:len(cqtbw) + 1] = cqtbw
    bw[len(cqtbw) + 1] = fbas[fbas_len + 2] - fbas[fbas_len]
    bw[-len(cqtbw):] = cqtbw[::-1]
    bw = bw / fftres
    fbas = fbas / fftres

    posit = np.zeros_like(fbas)
    posit[:fbas_len + 2] = np.floor(fbas[:fbas_len + 2])
    posit[fbas_len + 2:] = np.ceil(fbas[fbas_len + 2:])
    base_shift = -posit[-1] % signal_len
    shift = np.zeros_like(posit).astype("int32")
    shift[1:] = (posit[1:] - posit[:-1]).astype("int32")
    shift[0] = base_shift

    bw = np.round(bw)
    bwfac = 1
    M = bw

    min_win = 4
    for ii in range(len(bw)):
        if bw[ii] < min_win:
            bw[ii] = min_win
            M[ii] = bw[ii]

    def _win(numel):
        if numel % 2 == 0:
            s1 = np.arange(0, .5, 1. / numel)
            if len(s1) != numel // 2:
                # edge case with small floating point numbers...
                s1 = s1[:-1]
            s2 = np.arange(-.5, 0, 1. / numel)
            if len(s2) != numel // 2:
                # edge case with small floating point numbers...
                s2 = s2[:-1]
            x = np.concatenate((s1, s2))
        else:
            s1 = np.arange(0, .5, 1. / numel)
            s2 = np.arange(-.5 + .5 / numel, 0, 1. / numel)
            if len(s2) != numel // 2:  # assume integer truncate 27 // 2 = 13
                s2 = s2[:-1]
            x = np.concatenate((s1, s2))
        assert len(x) == numel
        g = .5 + .5 * np.cos(2 * np.pi * x)
        return g

    multiscale = [_win(bi) for bi in bw]
    bw = bwfac * np.ceil(M / bwfac)

    for kk in [0, fbas_len + 1]:
        if M[kk] > M[kk + 1]:
            multiscale[kk] = np.ones(M[kk]).astype(multiscale[0].dtype)
            i1 = np.floor(M[kk] / 2) - np.floor(M[kk + 1] / 2)
            i2 = np.floor(M[kk] / 2) + np.ceil(M[kk + 1] / 2)
            # Very rarely, gets an off by 1 error? Seems to be at the end...
            # for now, slice
            multiscale[kk][i1:i2] = _win(M[kk + 1])
            multiscale[kk] = multiscale[kk] / np.sqrt(M[kk])
    return multiscale, shift, M


def nsgtf_real(X, multiscale, shift, window_lens):
    """
    Nonstationary Gabor Transform for real values

    References
    ----------
    Velasco G. A., Holighaus N., Dorfler M., Grill T.
    Constructing an invertible constant-Q transform with nonstationary Gabor
    frames, Proceedings of the 14th International Conference on Digital
    Audio Effects (DAFx 11), Paris, France, 2011

    Holighaus N., Dorfler M., Velasco G. A. and Grill T.
    A framework for invertible, real-time constant-Q transforms, submitted.

    Original matlab code copyright follows:

    AUTHOR(s) : Monika Dorfler, Gino Angelo Velasco, Nicki Holighaus, 2010-2011

    COPYRIGHT : (c) NUHAG, Dept.Math., University of Vienna, AUSTRIA
    http://nuhag.eu/
    Permission is granted to modify and re-distribute this
    code in any manner as long as this notice is preserved.
    All standard disclaimers apply.
    """
    # This will break with multchannel input
    signal_len = len(X)
    N = len(shift)
    X_fft = np.fft.fft(X)

    fill = np.sum(shift) - signal_len
    if fill > 0:
        X_fft_tmp = np.zeros((signal_len + shift))
        X_fft_tmp[:len(X_fft)] = X_fft
        X_fft = X_fft_tmp
    posit = np.cumsum(shift) - shift[0]
    scale_lens = np.array([len(m) for m in multiscale])
    N = np.where(posit - np.floor(scale_lens) <= (signal_len + fill) / 2)[0][-1]
    c = []
    # c[0] is almost exact
    for ii in range(N):
        idx_l = np.arange(np.ceil(scale_lens[ii] / 2), scale_lens[ii])
        idx_r = np.arange(np.ceil(scale_lens[ii] / 2))
        idx = np.concatenate((idx_l, idx_r))
        idx = idx.astype("int32")
        subwin_range = posit[ii] + np.arange(-np.floor(scale_lens[ii] / 2),
                                             np.ceil(scale_lens[ii] / 2))
        win_range = subwin_range % (signal_len + fill)
        win_range = win_range.astype("int32")
        if window_lens[ii] < scale_lens[ii]:
            raise ValueError("Not handling 'not enough channels' case")
        else:
            temp = np.zeros((window_lens[ii],)).astype(X_fft.dtype)
            temp_idx_l = np.arange(len(temp) - np.floor(scale_lens[ii] / 2),
                                   len(temp))
            temp_idx_r = np.arange(np.ceil(scale_lens[ii] / 2))
            temp_idx = np.concatenate((temp_idx_l, temp_idx_r))
            temp_idx = temp_idx.astype("int32")
            temp[temp_idx] = X_fft[win_range] * multiscale[ii][idx]
            fs_new_bins = window_lens[ii]
            fk_bins = posit[ii]
            displace = fk_bins - np.floor(fk_bins / fs_new_bins) * fs_new_bins
            displace = displace.astype("int32")
            temp = np.roll(temp, displace)
        c.append(np.fft.ifft(temp))

    if 0:
        # cell2mat concatenation
        c = np.concatenate(c)
    return c


def nsdual(multiscale, shift, window_lens):
    """
    Calculation of nonstationary inverse gabor filters

    References
    ----------
    Velasco G. A., Holighaus N., Dorfler M., Grill T.
    Constructing an invertible constant-Q transform with nonstationary Gabor
    frames, Proceedings of the 14th International Conference on Digital
    Audio Effects (DAFx 11), Paris, France, 2011

    Holighaus N., Dorfler M., Velasco G. A. and Grill T.
    A framework for invertible, real-time constant-Q transforms, submitted.

    Original matlab code copyright follows:

    AUTHOR(s) : Monika Dorfler, Gino Angelo Velasco, Nicki Holighaus, 2010-2011

    COPYRIGHT : (c) NUHAG, Dept.Math., University of Vienna, AUSTRIA
    http://nuhag.eu/
    Permission is granted to modify and re-distribute this
    code in any manner as long as this notice is preserved.
    All standard disclaimers apply.
    """
    N = len(shift)
    posit = np.cumsum(shift)
    seq_len = posit[-1]
    posit = posit - shift[0]

    diagonal = np.zeros((seq_len,))
    win_range = []

    for ii in range(N):
        filt_len = len(multiscale[ii])
        idx = np.arange(-np.floor(filt_len / 2), np.ceil(filt_len / 2))
        win_range.append((posit[ii] + idx) % seq_len)
        subdiag = window_lens[ii] * np.fft.fftshift(multiscale[ii]) ** 2
        ind = win_range[ii].astype(np.int)
        diagonal[ind] = diagonal[ind] + subdiag

    dual_multiscale = multiscale
    for ii in range(N):
        ind = win_range[ii].astype(np.int)
        dual_multiscale[ii] = np.fft.ifftshift(
            np.fft.fftshift(dual_multiscale[ii]) / diagonal[ind])
    return dual_multiscale


def nsgitf_real(c, c_dc, c_nyq, multiscale, shift):
    """
    Nonstationary Inverse Gabor Transform on real valued signal

    References
    ----------
    Velasco G. A., Holighaus N., Dorfler M., Grill T.
    Constructing an invertible constant-Q transform with nonstationary Gabor
    frames, Proceedings of the 14th International Conference on Digital
    Audio Effects (DAFx 11), Paris, France, 2011

    Holighaus N., Dorfler M., Velasco G. A. and Grill T.
    A framework for invertible, real-time constant-Q transforms, submitted.

    Original matlab code copyright follows:

    AUTHOR(s) : Monika Dorfler, Gino Angelo Velasco, Nicki Holighaus, 2010-2011

    COPYRIGHT : (c) NUHAG, Dept.Math., University of Vienna, AUSTRIA
    http://nuhag.eu/
    Permission is granted to modify and re-distribute this
    code in any manner as long as this notice is preserved.
    All standard disclaimers apply.
    """
    c_l = []
    c_l.append(c_dc)
    c_l.extend([ci for ci in c])
    c_l.append(c_nyq)

    posit = np.cumsum(shift)
    seq_len = posit[-1]
    posit -= shift[0]
    out = np.zeros((seq_len,)).astype(c_l[1].dtype)

    for ii in range(len(c_l)):
        filt_len = len(multiscale[ii])
        win_range = posit[ii] + np.arange(-np.floor(filt_len / 2),
                                          np.ceil(filt_len / 2))
        win_range = (win_range % seq_len).astype(np.int)
        temp = np.fft.fft(c_l[ii]) * len(c_l[ii])

        fs_new_bins = len(c_l[ii])
        fk_bins = posit[ii]
        displace = int(fk_bins - np.floor(fk_bins / fs_new_bins) * fs_new_bins)
        temp = np.roll(temp, -displace)
        l = np.arange(len(temp) - np.floor(filt_len / 2), len(temp))
        r = np.arange(np.ceil(filt_len / 2))
        temp_idx = (np.concatenate((l, r)) % len(temp)).astype(np.int)
        temp = temp[temp_idx]
        lf = np.arange(filt_len - np.floor(filt_len / 2), filt_len)
        rf = np.arange(np.ceil(filt_len / 2))
        filt_idx = np.concatenate((lf, rf)).astype(np.int)
        m = multiscale[ii][filt_idx]
        out[win_range] = out[win_range] + m * temp

    nyq_bin = np.floor(seq_len / 2) + 1
    out_idx = np.arange(
        nyq_bin - np.abs(1 - seq_len % 2) - 1, 0, -1).astype(np.int)
    out[nyq_bin:] = np.conj(out[out_idx])
    t_out = np.real(np.fft.ifft(out)).astype(np.float64)
    return t_out


def cqt(X, fs, n_bins=48, fmin=27.5, fmax="nyq", gamma=20):
    """
    Constant Q Transform

    References
    ----------
    Velasco G. A., Holighaus N., Dorfler M., Grill T.
    Constructing an invertible constant-Q transform with nonstationary Gabor
    frames, Proceedings of the 14th International Conference on Digital
    Audio Effects (DAFx 11), Paris, France, 2011

    Holighaus N., Dorfler M., Velasco G. A. and Grill T.
    A framework for invertible, real-time constant-Q transforms, submitted.

    Original matlab code copyright follows:

    AUTHOR(s) : Monika Dorfler, Gino Angelo Velasco, Nicki Holighaus, 2010-2011

    COPYRIGHT : (c) NUHAG, Dept.Math., University of Vienna, AUSTRIA
    http://nuhag.eu/
    Permission is granted to modify and re-distribute this
    code in any manner as long as this notice is preserved.
    All standard disclaimers apply.
    """
    if fmax == "nyq":
        fmax = fs / 2.
    multiscale, shift, window_lens = nsgcwin(fmin, fmax, n_bins, fs,
                                             len(X), gamma)
    fbas = fs * np.cumsum(shift[1:]) / len(X)
    fbas = fbas[:len(window_lens) // 2 - 1]
    bins = window_lens.shape[0] // 2 - 1
    window_lens[1:bins + 1] = window_lens[bins + 2]
    window_lens[bins + 2:] = window_lens[1:bins + 1][::-1]
    norm = 2. * window_lens[:bins + 2] / float(len(X))
    norm = np.concatenate((norm, norm[1:-1][::-1]))
    multiscale = [norm[ii] * multiscale[ii] for ii in range(2 * (bins + 1))]

    c = nsgtf_real(X, multiscale, shift, window_lens)
    c_dc = c[0]
    c_nyq = c[-1]
    c_sub = c[1:-1]
    c = np.vstack(c_sub)
    return c, c_dc, c_nyq, multiscale, shift, window_lens


def icqt(X_cq, c_dc, c_nyq, multiscale, shift, window_lens):
    """
    Inverse constant Q Transform

    References
    ----------
    Velasco G. A., Holighaus N., Dorfler M., Grill T.
    Constructing an invertible constant-Q transform with nonstationary Gabor
    frames, Proceedings of the 14th International Conference on Digital
    Audio Effects (DAFx 11), Paris, France, 2011

    Holighaus N., Dorfler M., Velasco G. A. and Grill T.
    A framework for invertible, real-time constant-Q transforms, submitted.

    Original matlab code copyright follows:

    AUTHOR(s) : Monika Dorfler, Gino Angelo Velasco, Nicki Holighaus, 2010-2011

    COPYRIGHT : (c) NUHAG, Dept.Math., University of Vienna, AUSTRIA
    http://nuhag.eu/
    Permission is granted to modify and re-distribute this
    code in any manner as long as this notice is preserved.
    All standard disclaimers apply.
    """
    new_multiscale = nsdual(multiscale, shift, window_lens)
    X = nsgitf_real(X_cq, c_dc, c_nyq, new_multiscale, shift)
    return X


def rolling_mean(X, window_size):
    w = 1.0 / window_size * np.ones((window_size))
    return np.correlate(X, w, 'valid')


def rolling_window(X, window_size):
    # for 1d data
    shape = X.shape[:-1] + (X.shape[-1] - window_size + 1, window_size)
    strides = X.strides + (X.strides[-1],)
    return np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)


def voiced_unvoiced(X, window_size=256, window_step=128, copy=True):
    """
    Voiced unvoiced detection from a raw signal

    Based on code from:
        https://www.clear.rice.edu/elec532/PROJECTS96/lpc/code.html

    Other references:
        http://www.seas.ucla.edu/spapl/code/harmfreq_MOLRT_VAD.m

    Parameters
    ----------
    X : ndarray
        Raw input signal

    window_size : int, optional (default=256)
        The window size to use, in samples.

    window_step : int, optional (default=128)
        How far the window steps after each calculation, in samples.

    copy : bool, optional (default=True)
        Whether to make a copy of the input array or allow in place changes.
    """
    X = np.array(X, copy=copy)
    if len(X.shape) < 2:
        X = X[None]
    n_points = X.shape[1]
    n_windows = n_points // window_step
    # Padding
    pad_sizes = [(window_size - window_step) // 2,
                 window_size - window_step // 2]
    # TODO: Handling for odd window sizes / steps
    X = np.hstack((np.zeros((X.shape[0], pad_sizes[0])), X,
                   np.zeros((X.shape[0], pad_sizes[1]))))

    clipping_factor = 0.6
    b, a = sg.butter(10, np.pi * 9 / 40)
    voiced_unvoiced = np.zeros((n_windows, 1))
    period = np.zeros((n_windows, 1))
    for window in range(max(n_windows - 1, 1)):
        XX = X.ravel()[window * window_step + np.arange(window_size)]
        XX *= sg.hamming(len(XX))
        XX = sg.lfilter(b, a, XX)
        left_max = np.max(np.abs(XX[:len(XX) // 3]))
        right_max = np.max(np.abs(XX[-len(XX) // 3:]))
        clip_value = clipping_factor * np.min([left_max, right_max])
        XX_clip = np.clip(XX, clip_value, -clip_value)
        XX_corr = np.correlate(XX_clip, XX_clip, mode='full')
        center = np.argmax(XX_corr)
        right_XX_corr = XX_corr[center:]
        prev_window = max([window - 1, 0])
        if voiced_unvoiced[prev_window] > 0:
            # Want it to be harder to turn off than turn on
            strength_factor = .29
        else:
            strength_factor = .3
        start = np.where(right_XX_corr < .3 * XX_corr[center])[0]
        # 20 is hardcoded but should depend on samplerate?
        try:
            start = np.max([20, start[0]])
        except IndexError:
            start = 20
        search_corr = right_XX_corr[start:]
        index = np.argmax(search_corr)
        second_max = search_corr[index]
        if (second_max > strength_factor * XX_corr[center]):
            voiced_unvoiced[window] = 1
            period[window] = start + index - 1
        else:
            voiced_unvoiced[window] = 0
            period[window] = 0
    return np.array(voiced_unvoiced), np.array(period)


def lpc_analysis(X, order=8, window_step=128, window_size=2 * 128,
                 emphasis=0.9, voiced_start_threshold=.9,
                 voiced_stop_threshold=.6, truncate=False, copy=True):
    """
    Extract LPC coefficients from a signal

    Based on code from:
        http://labrosa.ee.columbia.edu/matlab/sws/

    _rParameters
    ----------
    X : ndarray
        Signals to extract LPC coefficients from

    order : int, optional (default=8)
        Order of the LPC coefficients. For speech, use the general rule that the
        order is two times the expected number of formants plus 2.
        This can be formulated as 2 + 2 * (fs // 2000). For approx. signals
        with fs = 7000, this is 8 coefficients - 2 + 2 * (7000 // 2000).

    window_step : int, optional (default=128)
        The size (in samples) of the space between each window

    window_size : int, optional (default=2 * 128)
        The size of each window (in samples) to extract coefficients over

    emphasis : float, optional (default=0.9)
        The emphasis coefficient to use for filtering

    voiced_start_threshold : float, optional (default=0.9)
        Upper power threshold for estimating when speech has started

    voiced_stop_threshold : float, optional (default=0.6)
        Lower power threshold for estimating when speech has stopped

    truncate : bool, optional (default=False)
        Whether to cut the data at the last window or do zero padding.

    copy : bool, optional (default=True)
        Whether to copy the input X or modify in place

    Returns
    -------
    lp_coefficients : ndarray
        lp coefficients to describe the frame

    per_frame_gain : ndarray
        calculated gain for each frame

    residual_excitation : ndarray
        leftover energy which is not described by lp coefficents and gain

    voiced_frames : ndarray
        array of [0, 1] values which holds voiced/unvoiced decision for each
        frame.

    References
    ----------
    D. P. W. Ellis (2004), "Sinewave Speech Analysis/Synthesis in Matlab",
    Web resource, available: http://www.ee.columbia.edu/ln/labrosa/matlab/sws/
    """
    X = np.array(X, copy=copy)
    if len(X.shape) < 2:
        X = X[None]

    n_points = X.shape[1]
    n_windows = int(n_points // window_step)
    if not truncate:
        pad_sizes = [(window_size - window_step) // 2,
                     window_size - window_step // 2]
        # TODO: Handling for odd window sizes / steps
        X = np.hstack((np.zeros((X.shape[0], pad_sizes[0])), X,
                       np.zeros((X.shape[0], pad_sizes[1]))))
    else:
        pad_sizes = [0, 0]
        X = X[0, :n_windows * window_step]

    lp_coefficients = np.zeros((n_windows, order + 1))
    per_frame_gain = np.zeros((n_windows, 1))
    residual_excitation = np.zeros(
        ((n_windows - 1) * window_step + window_size))
    # Pre-emphasis high-pass filter
    X = sg.lfilter([1, -emphasis], 1, X)
    # stride_tricks.as_strided?
    autocorr_X = np.zeros((n_windows, 2 * window_size - 1))
    for window in range(max(n_windows - 1, 1)):
        wtws = int(window * window_step)
        XX = X.ravel()[wtws + np.arange(window_size, dtype="int32")]
        WXX = XX * sg.hanning(window_size)
        autocorr_X[window] = np.correlate(WXX, WXX, mode='full')
        center = np.argmax(autocorr_X[window])
        RXX = autocorr_X[window,
                         np.arange(center, window_size + order, dtype="int32")]
        R = linalg.toeplitz(RXX[:-1])
        solved_R = linalg.pinv(R).dot(RXX[1:])
        filter_coefs = np.hstack((1, -solved_R))
        residual_signal = sg.lfilter(filter_coefs, 1, WXX)
        gain = np.sqrt(np.mean(residual_signal ** 2))
        lp_coefficients[window] = filter_coefs
        per_frame_gain[window] = gain
        assign_range = wtws + np.arange(window_size, dtype="int32")
        residual_excitation[assign_range] += residual_signal / gain
    # Throw away first part in overlap mode for proper synthesis
    residual_excitation = residual_excitation[pad_sizes[0]:]
    return lp_coefficients, per_frame_gain, residual_excitation


def lpc_to_frequency(lp_coefficients, per_frame_gain):
    """
    Extract resonant frequencies and magnitudes from LPC coefficients and gains.
    Parameters
    ----------
    lp_coefficients : ndarray
        LPC coefficients, such as those calculated by ``lpc_analysis``

    per_frame_gain : ndarray
       Gain calculated for each frame, such as those calculated
       by ``lpc_analysis``

    Returns
    -------
    frequencies : ndarray
       Resonant frequencies calculated from LPC coefficients and gain. Returned
       frequencies are from 0 to 2 * pi

    magnitudes : ndarray
       Magnitudes of resonant frequencies

    References
    ----------
    D. P. W. Ellis (2004), "Sinewave Speech Analysis/Synthesis in Matlab",
    Web resource, available: http://www.ee.columbia.edu/ln/labrosa/matlab/sws/
    """
    n_windows, order = lp_coefficients.shape

    frame_frequencies = np.zeros((n_windows, (order - 1) // 2))
    frame_magnitudes = np.zeros_like(frame_frequencies)

    for window in range(n_windows):
        w_coefs = lp_coefficients[window]
        g_coefs = per_frame_gain[window]
        roots = np.roots(np.hstack(([1], w_coefs[1:])))
        # Roots doesn't return the same thing as MATLAB... agh
        frequencies, index = np.unique(
            np.abs(np.angle(roots)), return_index=True)
        # Make sure 0 doesn't show up...
        gtz = np.where(frequencies > 0)[0]
        frequencies = frequencies[gtz]
        index = index[gtz]
        magnitudes = g_coefs / (1. - np.abs(roots))
        sort_index = np.argsort(frequencies)
        frame_frequencies[window, :len(sort_index)] = frequencies[sort_index]
        frame_magnitudes[window, :len(sort_index)] = magnitudes[sort_index]
    return frame_frequencies, frame_magnitudes


def lpc_to_lsf(all_lpc):
    if len(all_lpc.shape) < 2:
        all_lpc = all_lpc[None]
    order = all_lpc.shape[1] - 1
    all_lsf = np.zeros((len(all_lpc), order))
    for i in range(len(all_lpc)):
        lpc = all_lpc[i]
        lpc1 = np.append(lpc, 0)
        lpc2 = lpc1[::-1]
        sum_filt = lpc1 + lpc2
        diff_filt = lpc1 - lpc2

        if order % 2 != 0:
            deconv_diff, _ = sg.deconvolve(diff_filt, [1, 0, -1])
            deconv_sum = sum_filt
        else:
            deconv_diff, _ = sg.deconvolve(diff_filt, [1, -1])
            deconv_sum, _ = sg.deconvolve(sum_filt, [1, 1])

        roots_diff = np.roots(deconv_diff)
        roots_sum = np.roots(deconv_sum)
        angle_diff = np.angle(roots_diff[::2])
        angle_sum = np.angle(roots_sum[::2])
        lsf = np.sort(np.hstack((angle_diff, angle_sum)))
        if len(lsf) != 0:
            all_lsf[i] = lsf
    return np.squeeze(all_lsf)


def lsf_to_lpc(all_lsf):
    if len(all_lsf.shape) < 2:
        all_lsf = all_lsf[None]
    order = all_lsf.shape[1]
    all_lpc = np.zeros((len(all_lsf), order + 1))
    for i in range(len(all_lsf)):
        lsf = all_lsf[i]
        zeros = np.exp(1j * lsf)
        sum_zeros = zeros[::2]
        diff_zeros = zeros[1::2]
        sum_zeros = np.hstack((sum_zeros, np.conj(sum_zeros)))
        diff_zeros = np.hstack((diff_zeros, np.conj(diff_zeros)))
        sum_filt = np.poly(sum_zeros)
        diff_filt = np.poly(diff_zeros)

        if order % 2 != 0:
            deconv_diff = sg.convolve(diff_filt, [1, 0, -1])
            deconv_sum = sum_filt
        else:
            deconv_diff = sg.convolve(diff_filt, [1, -1])
            deconv_sum = sg.convolve(sum_filt, [1, 1])

        lpc = .5 * (deconv_sum + deconv_diff)
        # Last coefficient is 0 and not returned
        all_lpc[i] = lpc[:-1]
    return np.squeeze(all_lpc)


def lpc_synthesis(lp_coefficients, per_frame_gain, residual_excitation=None,
                  voiced_frames=None, window_step=128, emphasis=0.9):
    """
    Synthesize a signal from LPC coefficients

    Based on code from:
        http://labrosa.ee.columbia.edu/matlab/sws/
        http://web.uvic.ca/~tyoon/resource/auditorytoolbox/auditorytoolbox/synlpc.html

    Parameters
    ----------
    lp_coefficients : ndarray
        Linear prediction coefficients

    per_frame_gain : ndarray
        Gain coefficients

    residual_excitation : ndarray or None, optional (default=None)
        Residual excitations. If None, this will be synthesized with white noise

    voiced_frames : ndarray or None, optional (default=None)
        Voiced frames. If None, all frames assumed to be voiced.

    window_step : int, optional (default=128)
        The size (in samples) of the space between each window

    emphasis : float, optional (default=0.9)
        The emphasis coefficient to use for filtering

    overlap_add : bool, optional (default=True)
        What type of processing to use when joining windows

    copy : bool, optional (default=True)
       Whether to copy the input X or modify in place

    Returns
    -------
    synthesized : ndarray
        Sound vector synthesized from input arguments

    References
    ----------
    D. P. W. Ellis (2004), "Sinewave Speech Analysis/Synthesis in Matlab",
    Web resource, available: http://www.ee.columbia.edu/ln/labrosa/matlab/sws/
    """
    # TODO: Incorporate better synthesis from
    # http://eecs.oregonstate.edu/education/docs/ece352/CompleteManual.pdf
    window_size = 2 * window_step
    [n_windows, order] = lp_coefficients.shape

    n_points = (n_windows + 1) * window_step
    n_excitation_points = n_points + window_step + window_step // 2

    random_state = np.random.RandomState(1999)
    if residual_excitation is None:
        # Need to generate excitation
        if voiced_frames is None:
            # No voiced/unvoiced info
            voiced_frames = np.ones((lp_coefficients.shape[0], 1))
        residual_excitation = np.zeros((n_excitation_points))
        f, m = lpc_to_frequency(lp_coefficients, per_frame_gain)
        t = np.linspace(0, 1, window_size, endpoint=False)
        hanning = sg.hanning(window_size)
        for window in range(n_windows):
            window_base = window * window_step
            index = window_base + np.arange(window_size)
            if voiced_frames[window]:
                sig = np.zeros_like(t)
                cycles = np.cumsum(f[window][0] * t)
                sig += sg.sawtooth(cycles, 0.001)
                residual_excitation[index] += hanning * sig
            residual_excitation[index] += hanning * 0.01 * random_state.randn(
                window_size)
    else:
        n_excitation_points = residual_excitation.shape[0]
        n_points = n_excitation_points + window_step + window_step // 2
    residual_excitation = np.hstack((residual_excitation,
                                     np.zeros(window_size)))
    if voiced_frames is None:
        voiced_frames = np.ones_like(per_frame_gain)

    synthesized = np.zeros((n_points))
    for window in range(n_windows):
        window_base = window * window_step
        oldbit = synthesized[window_base + np.arange(window_step)]
        w_coefs = lp_coefficients[window]
        if not np.all(w_coefs):
            # Hack to make lfilter avoid
            # ValueError: BUG: filter coefficient a[0] == 0 not supported yet
            # when all coeffs are 0
            w_coefs = [1]
        g_coefs = voiced_frames[window] * per_frame_gain[window]
        index = window_base + np.arange(window_size)
        newbit = g_coefs * sg.lfilter([1], w_coefs,
                                      residual_excitation[index])
        synthesized[index] = np.hstack((oldbit, np.zeros(
            (window_size - window_step))))
        synthesized[index] += sg.hanning(window_size) * newbit
    synthesized = sg.lfilter([1], [1, -emphasis], synthesized)
    return synthesized


def soundsc(X, gain_scale=.9, copy=True):
    """
    Approximate implementation of soundsc from MATLAB without the audio playing.

    Parameters
    ----------
    X : ndarray
        Signal to be rescaled

    gain_scale : float
        Gain multipler, default .9 (90% of maximum representation)

    copy : bool, optional (default=True)
        Whether to make a copy of input signal or operate in place.

    Returns
    -------
    X_sc : ndarray
        (-32767, 32767) scaled version of X as int16, suitable for writing
        with scipy.io.wavfile
    """
    X = np.array(X, copy=copy)
    X = (X - X.min()) / (X.max() - X.min())
    X = 2 * X - 1
    X = gain_scale * X
    X = X * 2 ** 15
    return X.astype('int16')


def _wav2array(nchannels, sampwidth, data):
    # wavio.py
    # Author: Warren Weckesser
    # License: BSD 3-Clause (http://opensource.org/licenses/BSD-3-Clause)

    """data must be the string containing the bytes from the wav file."""
    num_samples, remainder = divmod(len(data), sampwidth * nchannels)
    if remainder > 0:
        raise ValueError('The length of data is not a multiple of '
                         'sampwidth * num_channels.')
    if sampwidth > 4:
        raise ValueError("sampwidth must not be greater than 4.")

    if sampwidth == 3:
        a = np.empty((num_samples, nchannels, 4), dtype=np.uint8)
        raw_bytes = np.fromstring(data, dtype=np.uint8)
        a[:, :, :sampwidth] = raw_bytes.reshape(-1, nchannels, sampwidth)
        a[:, :, sampwidth:] = (a[:, :, sampwidth - 1:sampwidth] >> 7) * 255
        result = a.view('<i4').reshape(a.shape[:-1])
    else:
        # 8 bit samples are stored as unsigned ints; others as signed ints.
        dt_char = 'u' if sampwidth == 1 else 'i'
        a = np.fromstring(data, dtype='<%s%d' % (dt_char, sampwidth))
        result = a.reshape(-1, nchannels)
    return result


def readwav(file):
    # wavio.py
    # Author: Warren Weckesser
    # License: BSD 3-Clause (http://opensource.org/licenses/BSD-3-Clause)
    """
    Read a wav file.

    Returns the frame rate, sample width (in bytes) and a numpy array
    containing the data.

    This function does not read compressed wav files.
    """
    wav = wave.open(file)
    rate = wav.getframerate()
    nchannels = wav.getnchannels()
    sampwidth = wav.getsampwidth()
    nframes = wav.getnframes()
    data = wav.readframes(nframes)
    wav.close()
    array = _wav2array(nchannels, sampwidth, data)
    return rate, sampwidth, array


def csvd(arr):
    """
    Do the complex SVD of a 2D array, returning real valued U, S, VT

    http://stemblab.github.io/complex-svd/
    """
    C_r = arr.real
    C_i = arr.imag
    block_x = C_r.shape[0]
    block_y = C_r.shape[1]
    K = np.zeros((2 * block_x, 2 * block_y))
    # Upper left
    K[:block_x, :block_y] = C_r
    # Lower left
    K[:block_x, block_y:] = C_i
    # Upper right
    K[block_x:, :block_y] = -C_i
    # Lower right
    K[block_x:, block_y:] = C_r
    return svd(K, full_matrices=False)


def icsvd(U, S, VT):
    """
    Invert back to complex values from the output of csvd

    U, S, VT = csvd(X)
    X_rec = inv_csvd(U, S, VT)
    """
    K = U.dot(np.diag(S)).dot(VT)
    block_x = U.shape[0] // 2
    block_y = U.shape[1] // 2
    arr_rec = np.zeros((block_x, block_y)) + 0j
    arr_rec.real = K[:block_x, :block_y]
    arr_rec.imag = K[:block_x, block_y:]
    return arr_rec


def sinusoid_analysis(X, input_sample_rate, resample_block=128, copy=True):
    """
    Contruct a sinusoidal model for the input signal.

    Parameters
    ----------
    X : ndarray
        Input signal to model

    input_sample_rate : int
        The sample rate of the input signal

    resample_block : int, optional (default=128)
       Controls the step size of the sinusoidal model

    Returns
    -------
    frequencies_hz : ndarray
       Frequencies for the sinusoids, in Hz.

    magnitudes : ndarray
       Magnitudes of sinusoids returned in ``frequencies``

    References
    ----------
    D. P. W. Ellis (2004), "Sinewave Speech Analysis/Synthesis in Matlab",
    Web resource, available: http://www.ee.columbia.edu/ln/labrosa/matlab/sws/
    """
    X = np.array(X, copy=copy)
    resample_to = 8000
    if input_sample_rate != resample_to:
        if input_sample_rate % resample_to != 0:
            raise ValueError("Input sample rate must be a multiple of 8k!")
        # Should be able to use resample... ?
        # resampled_count = round(len(X) * resample_to / input_sample_rate)
        # X = sg.resample(X, resampled_count, window=sg.hanning(len(X)))
        X = sg.decimate(X, input_sample_rate // resample_to)
    step_size = 2 * round(resample_block / input_sample_rate * resample_to / 2.)
    a, g, e = lpc_analysis(X, order=8, window_step=step_size,
                           window_size=2 * step_size)
    f, m = lpc_to_frequency(a, g)
    f_hz = f * resample_to / (2 * np.pi)
    return f_hz, m


def slinterp(X, factor, copy=True):
    """
    Slow-ish linear interpolation of a 1D numpy array. There must be some
    better function to do this in numpy.

    Parameters
    ----------
    X : ndarray
        1D input array to interpolate

    factor : int
        Integer factor to interpolate by

    Return
    ------
    X_r : ndarray
    """
    sz = np.product(X.shape)
    X = np.array(X, copy=copy)
    X_s = np.hstack((X[1:], [0]))
    X_r = np.zeros((factor, sz))
    for i in range(factor):
        X_r[i, :] = (factor - i) / float(factor) * X + (i / float(factor)) * X_s
    return X_r.T.ravel()[:(sz - 1) * factor + 1]


def sinusoid_synthesis(frequencies_hz, magnitudes, input_sample_rate=16000,
                       resample_block=128):
    """
    Create a time series based on input frequencies and magnitudes.

    Parameters
    ----------
    frequencies_hz : ndarray
        Input signal to model

    magnitudes : int
        The sample rate of the input signal

    input_sample_rate : int, optional (default=16000)
        The sample rate parameter that the sinusoid analysis was run with

    resample_block : int, optional (default=128)
       Controls the step size of the sinusoidal model

    Returns
    -------
    synthesized : ndarray
        Sound vector synthesized from input arguments

    References
    ----------
    D. P. W. Ellis (2004), "Sinewave Speech Analysis/Synthesis in Matlab",
    Web resource, available: http://www.ee.columbia.edu/ln/labrosa/matlab/sws/
    """
    rows, cols = frequencies_hz.shape
    synthesized = np.zeros((1 + ((rows - 1) * resample_block),))
    for col in range(cols):
        mags = slinterp(magnitudes[:, col], resample_block)
        freqs = slinterp(frequencies_hz[:, col], resample_block)
        cycles = np.cumsum(2 * np.pi * freqs / float(input_sample_rate))
        sines = mags * np.cos(cycles)
        synthesized += sines
    return synthesized


def compress(X, n_components, window_size=128):
    """
    Compress using the DCT

    Parameters
    ----------
    X : ndarray, shape=(n_samples,)
        The input signal to compress. Should be 1-dimensional

    n_components : int
        The number of DCT components to keep. Setting n_components to about
        .5 * window_size can give compression with fairly good reconstruction.

    window_size : int
        The input X is broken into windows of window_size, each of which are
        then compressed with the DCT.

    Returns
    -------
    X_compressed : ndarray, shape=(num_windows, window_size)
       A 2D array of non-overlapping DCT coefficients. For use with uncompress

    Reference
    ---------
    http://nbviewer.ipython.org/github/craffel/crucialpython/blob/master/week3/stride_tricks.ipynb
    """
    if len(X) % window_size != 0:
        append = np.zeros((window_size - len(X) % window_size))
        X = np.hstack((X, append))
    num_frames = len(X) // window_size
    X_strided = X.reshape((num_frames, window_size))
    X_dct = fftpack.dct(X_strided, norm='ortho')
    if n_components is not None:
        X_dct = X_dct[:, :n_components]
    return X_dct


def uncompress(X_compressed, window_size=128):
    """
    Uncompress a DCT compressed signal (such as returned by ``compress``).

    Parameters
    ----------
    X_compressed : ndarray, shape=(n_samples, n_features)
        Windowed and compressed array.

    window_size : int, optional (default=128)
        Size of the window used when ``compress`` was called.

    Returns
    -------
    X_reconstructed : ndarray, shape=(n_samples)
        Reconstructed version of X.
    """
    if X_compressed.shape[1] % window_size != 0:
        append = np.zeros((X_compressed.shape[0],
                           window_size - X_compressed.shape[1] % window_size))
        X_compressed = np.hstack((X_compressed, append))
    X_r = fftpack.idct(X_compressed, norm='ortho')
    return X_r.ravel()


def sine_window(X):
    """
    Apply a sinusoid window to X.

    Parameters
    ----------
    X : ndarray, shape=(n_samples, n_features)
        Input array of samples

    Returns
    -------
    X_windowed : ndarray, shape=(n_samples, n_features)
        Windowed version of X.
    """
    i = np.arange(X.shape[1])
    win = np.sin(np.pi * (i + 0.5) / X.shape[1])
    row_stride = 0
    col_stride = win.itemsize
    strided_win = as_strided(win, shape=X.shape,
                             strides=(row_stride, col_stride))
    return X * strided_win


def kaiserbessel_window(X, alpha=6.5):
    """
    Apply a Kaiser-Bessel window to X.

    Parameters
    ----------
    X : ndarray, shape=(n_samples, n_features)
        Input array of samples

    alpha : float, optional (default=6.5)
        Tuning parameter for Kaiser-Bessel function. alpha=6.5 should make
        perfect reconstruction possible for DCT.

    Returns
    -------
    X_windowed : ndarray, shape=(n_samples, n_features)
        Windowed version of X.
    """
    beta = np.pi * alpha
    win = sg.kaiser(X.shape[1], beta)
    row_stride = 0
    col_stride = win.itemsize
    strided_win = as_strided(win, shape=X.shape,
                             strides=(row_stride, col_stride))
    return X * strided_win


def overlap(X, window_size, window_step):
    """
    Create an overlapped version of X

    Parameters
    ----------
    X : ndarray, shape=(n_samples,)
        Input signal to window and overlap

    window_size : int
        Size of windows to take

    window_step : int
        Step size between windows

    Returns
    -------
    X_strided : shape=(n_windows, window_size)
        2D array of overlapped X
    """
    if window_size % 2 != 0:
        raise ValueError("Window size must be even!")
    # Make sure there are an even number of windows before stridetricks
    append = np.zeros((window_size - len(X) % window_size))
    X = np.hstack((X, append))
    overlap_sz = window_size - window_step
    new_shape = X.shape[:-1] + ((X.shape[-1] - overlap_sz) // window_step, window_size)
    new_strides = X.strides[:-1] + (window_step * X.strides[-1],) + X.strides[-1:]
    X_strided = as_strided(X, shape=new_shape, strides=new_strides)
    return X_strided


def halfoverlap(X, window_size):
    """
    Create an overlapped version of X using 50% of window_size as overlap.

    Parameters
    ----------
    X : ndarray, shape=(n_samples,)
        Input signal to window and overlap

    window_size : int
        Size of windows to take

    Returns
    -------
    X_strided : shape=(n_windows, window_size)
        2D array of overlapped X
    """
    if window_size % 2 != 0:
        raise ValueError("Window size must be even!")
    window_step = window_size // 2
    # Make sure there are an even number of windows before stridetricks
    append = np.zeros((window_size - len(X) % window_size))
    X = np.hstack((X, append))
    num_frames = len(X) // window_step - 1
    row_stride = X.itemsize * window_step
    col_stride = X.itemsize
    X_strided = as_strided(X, shape=(num_frames, window_size),
                           strides=(row_stride, col_stride))
    return X_strided


def invert_halfoverlap(X_strided):
    """
    Invert ``halfoverlap`` function to reconstruct X

    Parameters
    ----------
    X_strided : ndarray, shape=(n_windows, window_size)
        X as overlapped windows

    Returns
    -------
    X : ndarray, shape=(n_samples,)
        Reconstructed version of X
    """
    # Hardcoded 50% overlap! Can generalize later...
    n_rows, n_cols = X_strided.shape
    X = np.zeros((((int(n_rows // 2) + 1) * n_cols),)).astype(X_strided.dtype)
    start_index = 0
    end_index = n_cols
    window_step = n_cols // 2
    for row in range(X_strided.shape[0]):
        X[start_index:end_index] += X_strided[row]
        start_index += window_step
        end_index += window_step
    return X


def overlap_add(X_strided, window_step, wsola=False):
    """
    overlap add to reconstruct X

    Parameters
    ----------
    X_strided : ndarray, shape=(n_windows, window_size)
        X as overlapped windows

    window_step : int
       step size for overlap add

    Returns
    -------
    X : ndarray, shape=(n_samples,)
        Reconstructed version of X
    """
    n_rows, window_size = X_strided.shape

    # Start with largest size (no overlap) then truncate after we finish
    # +2 for one window on each side
    X = np.zeros(((n_rows + 2) * window_size,)).astype(X_strided.dtype)
    start_index = 0

    total_windowing_sum = np.zeros((X.shape[0]))
    win = 0.54 - .46 * np.cos(2 * np.pi * np.arange(window_size) / (
        window_size - 1))
    for i in range(n_rows):
        end_index = start_index + window_size
        if wsola:
            offset_size = window_size - window_step
            offset = xcorr_offset(X[start_index:start_index + offset_size],
                                  X_strided[i, :offset_size])
            ss = start_index - offset
            st = end_index - offset
            if start_index - offset < 0:
                ss = 0
                st = 0 + (end_index - start_index)
            X[ss:st] += X_strided[i]
            total_windowing_sum[ss:st] += win
            start_index = start_index + window_step
        else:
            X[start_index:end_index] += X_strided[i]
            total_windowing_sum[start_index:end_index] += win
            start_index += window_step
    # Not using this right now
    #X = np.real(X) / (total_windowing_sum + 1)
    X = X[:end_index]
    return X


def overlap_compress(X, n_components, window_size):
    """
    Overlap (at 50% of window_size) and compress X.

    Parameters
    ----------
    X : ndarray, shape=(n_samples,)
        Input signal to compress

    n_components : int
        number of DCT components to keep

    window_size : int
        Size of windows to take

    Returns
    -------
    X_dct : ndarray, shape=(n_windows, n_components)
        Windowed and compressed version of X
    """
    X_strided = halfoverlap(X, window_size)
    X_dct = fftpack.dct(X_strided, norm='ortho')
    if n_components is not None:
        X_dct = X_dct[:, :n_components]
    return X_dct


# Evil voice is caused by adding double the zeros before inverse DCT...
# Very cool bug but makes sense
def overlap_uncompress(X_compressed, window_size):
    """
    Uncompress X as returned from ``overlap_compress``.

    Parameters
    ----------
    X_compressed : ndarray, shape=(n_windows, n_components)
        Windowed and compressed version of X

    window_size : int
        Size of windows originally used when compressing X

    Returns
    -------
    X_reconstructed : ndarray, shape=(n_samples,)
        Reconstructed version of X
    """
    if X_compressed.shape[1] % window_size != 0:
        append = np.zeros((X_compressed.shape[0], window_size -
                           X_compressed.shape[1] % window_size))
        X_compressed = np.hstack((X_compressed, append))
    X_r = fftpack.idct(X_compressed, norm='ortho')
    return invert_halfoverlap(X_r)


def herz_to_mel(freqs):
    """
    Based on code by Dan Ellis

    http://labrosa.ee.columbia.edu/matlab/tf_agc/
    """
    f_0 = 0  # 133.33333
    f_sp = 200 / 3.  # 66.66667
    bark_freq = 1000.
    bark_pt = (bark_freq - f_0) / f_sp
    # The magic 1.0711703 which is the ratio needed to get from 1000 Hz
    # to 6400 Hz in 27 steps, and is *almost* the ratio between 1000 Hz
    # and the preceding linear filter center at 933.33333 Hz
    # (actually 1000/933.33333 = 1.07142857142857 and
    # exp(log(6.4)/27) = 1.07117028749447)
    if not isinstance(freqs, np.ndarray):
        freqs = np.array(freqs)[None]
    log_step = np.exp(np.log(6.4) / 27)
    lin_pts = (freqs < bark_freq)
    mel = 0. * freqs
    mel[lin_pts] = (freqs[lin_pts] - f_0) / f_sp
    mel[~lin_pts] = bark_pt + np.log(freqs[~lin_pts] / bark_freq) / np.log(
        log_step)
    return mel


def mel_to_herz(mel):
    """
    Based on code by Dan Ellis

    http://labrosa.ee.columbia.edu/matlab/tf_agc/
    """
    f_0 = 0  # 133.33333
    f_sp = 200 / 3.  # 66.66667
    bark_freq = 1000.
    bark_pt = (bark_freq - f_0) / f_sp
    # The magic 1.0711703 which is the ratio needed to get from 1000 Hz
    # to 6400 Hz in 27 steps, and is *almost* the ratio between 1000 Hz
    # and the preceding linear filter center at 933.33333 Hz
    # (actually 1000/933.33333 = 1.07142857142857 and
    # exp(log(6.4)/27) = 1.07117028749447)
    if not isinstance(mel, np.ndarray):
        mel = np.array(mel)[None]
    log_step = np.exp(np.log(6.4) / 27)
    lin_pts = (mel < bark_pt)

    freqs = 0. * mel
    freqs[lin_pts] = f_0 + f_sp * mel[lin_pts]
    freqs[~lin_pts] = bark_freq * np.exp(np.log(log_step) * (
        mel[~lin_pts] - bark_pt))
    return freqs


def mel_freq_weights(n_fft, fs, n_filts=None, width=None):
    """
    Based on code by Dan Ellis

    http://labrosa.ee.columbia.edu/matlab/tf_agc/
    """
    min_freq = 0
    max_freq = fs // 2
    if width is None:
        width = 1.
    if n_filts is None:
        n_filts = int(herz_to_mel(max_freq) / 2) + 1
    else:
        n_filts = int(n_filts)
        assert n_filts > 0
    weights = np.zeros((n_filts, n_fft))
    fft_freqs = np.arange(n_fft // 2) / n_fft * fs
    min_mel = herz_to_mel(min_freq)
    max_mel = herz_to_mel(max_freq)
    partial = np.arange(n_filts + 2) / (n_filts + 1.) * (max_mel - min_mel)
    bin_freqs = mel_to_herz(min_mel + partial)
    bin_bin = np.round(bin_freqs / fs * (n_fft - 1))
    for i in range(n_filts):
        fs_i = bin_freqs[i + np.arange(3)]
        fs_i = fs_i[1] + width * (fs_i - fs_i[1])
        lo_slope = (fft_freqs - fs_i[0]) / float(fs_i[1] - fs_i[0])
        hi_slope = (fs_i[2] - fft_freqs) / float(fs_i[2] - fs_i[1])
        weights[i, :n_fft // 2] = np.maximum(
            0, np.minimum(lo_slope, hi_slope))
    # Constant amplitude multiplier
    weights = np.diag(2. / (bin_freqs[2:n_filts + 2]
                      - bin_freqs[:n_filts])).dot(weights)
    weights[:, n_fft // 2:] = 0
    return weights


def time_attack_agc(X, fs, t_scale=0.5, f_scale=1.):
    """
    AGC based on code by Dan Ellis

    http://labrosa.ee.columbia.edu/matlab/tf_agc/
    """
    # 32 ms grid for FFT
    n_fft = 2 ** int(np.log(0.032 * fs) / np.log(2))
    f_scale = float(f_scale)
    window_size = n_fft
    window_step = window_size // 2
    X_freq = stft(X, window_size, mean_normalize=False)
    fft_fs = fs / window_step
    n_bands = max(10, 20 / f_scale)
    mel_width = f_scale * n_bands / 10.
    f_to_a = mel_freq_weights(n_fft, fs, n_bands, mel_width)
    f_to_a = f_to_a[:, :n_fft // 2 + 1]
    audiogram = np.abs(X_freq).dot(f_to_a.T)
    fbg = np.zeros_like(audiogram)
    state = np.zeros((audiogram.shape[1],))
    alpha = np.exp(-(1. / fft_fs) / t_scale)
    for i in range(len(audiogram)):
        state = np.maximum(alpha * state, audiogram[i])
        fbg[i] = state

    sf_to_a = np.sum(f_to_a, axis=0)
    E = np.diag(1. / (sf_to_a + (sf_to_a == 0)))
    E = E.dot(f_to_a.T)
    E = fbg.dot(E.T)
    E[E <= 0] = np.min(E[E > 0])
    ts = istft(X_freq / E, window_size, mean_normalize=False)
    return ts, X_freq, E


def hebbian_kmeans(X, n_clusters=10, n_epochs=10, W=None, learning_rate=0.01,
                   batch_size=100, random_state=None, verbose=True):
    """
    Modified from existing code from R. Memisevic
    See http://www.cs.toronto.edu/~rfm/code/hebbian_kmeans.py
    """
    if W is None:
        if random_state is None:
            random_state = np.random.RandomState()
        W = 0.1 * random_state.randn(n_clusters, X.shape[1])
    else:
        assert n_clusters == W.shape[0]
    X2 = (X ** 2).sum(axis=1, keepdims=True)
    last_print = 0
    for e in range(n_epochs):
        for i in range(0, X.shape[0], batch_size):
            X_i = X[i: i + batch_size]
            X2_i = X2[i: i + batch_size]
            D = -2 * np.dot(W, X_i.T)
            D += (W ** 2).sum(axis=1, keepdims=True)
            D += X2_i.T
            S = (D == D.min(axis=0)[None, :]).astype("float").T
            W += learning_rate * (
                np.dot(S.T, X_i) - S.sum(axis=0)[:, None] * W)
        if verbose:
            if e == 0 or e > (.05 * n_epochs + last_print):
                last_print = e
                print("Epoch %i of %i, cost %.4f" % (
                    e + 1, n_epochs, D.min(axis=0).sum()))
    return W


def complex_to_real_view(arr_c):
    # Inplace view from complex to r, i as separate columns
    assert arr_c.dtype in [np.complex64, np.complex128]
    shp = arr_c.shape
    dtype = np.float64 if arr_c.dtype == np.complex128 else np.float32
    arr_r = arr_c.ravel().view(dtype=dtype).reshape(shp[0], 2 * shp[1])
    return arr_r


def real_to_complex_view(arr_r):
    # Inplace view from real, image as columns to complex
    assert arr_r.dtype not in [np.complex64, np.complex128]
    shp = arr_r.shape
    dtype = np.complex128 if arr_r.dtype == np.float64 else np.complex64
    arr_c = arr_r.ravel().view(dtype=dtype).reshape(shp[0], shp[1] // 2)
    return arr_c


def complex_to_abs(arr_c):
    return np.abs(arr_c)


def complex_to_angle(arr_c):
    return np.angle(arr_c)


def abs_and_angle_to_complex(arr_abs, arr_angle):
    # abs(f_c2 - f_c) < 1E-15
    return arr_abs * np.exp(1j * arr_angle)


def angle_to_sin_cos(arr_angle):
    return np.hstack((np.sin(arr_angle), np.cos(arr_angle)))


def sin_cos_to_angle(arr_sin, arr_cos):
    return np.arctan2(arr_sin, arr_cos)


def polyphase_core(x, m, f):
    # x = input data
    # m = decimation rate
    # f = filter
    # Hack job - append zeros to match decimation rate
    if x.shape[0] % m != 0:
        x = np.append(x, np.zeros((m - x.shape[0] % m,)))
    if f.shape[0] % m != 0:
        f = np.append(f, np.zeros((m - f.shape[0] % m,)))
    polyphase = p = np.zeros((m, (x.shape[0] + f.shape[0]) / m), dtype=x.dtype)
    p[0, :-1] = np.convolve(x[::m], f[::m])
    # Invert the x values when applying filters
    for i in range(1, m):
        p[i, 1:] = np.convolve(x[m - i::m], f[i::m])
    return p


def polyphase_single_filter(x, m, f):
    return np.sum(polyphase_core(x, m, f), axis=0)


def polyphase_lowpass(arr, downsample=2, n_taps=50, filter_pad=1.1):
    filt = firwin(downsample * n_taps, 1 / (downsample * filter_pad))
    filtered = polyphase_single_filter(arr, downsample, filt)
    return filtered


def window(arr, window_size, window_step=1, axis=0):
    """
    Directly taken from Erik Rigtorp's post to numpy-discussion.
    <http://www.mail-archive.com/numpy-discussion@scipy.org/msg29450.html>

    <http://stackoverflow.com/questions/4936620/using-strides-for-an-efficient-moving-average-filter>
    """
    if window_size < 1:
        raise ValueError("`window_size` must be at least 1.")
    if window_size > arr.shape[-1]:
        raise ValueError("`window_size` is too long.")

    orig = list(range(len(arr.shape)))
    trans = list(range(len(arr.shape)))
    trans[axis] = orig[-1]
    trans[-1] = orig[axis]
    arr = arr.transpose(trans)

    shape = arr.shape[:-1] + (arr.shape[-1] - window_size + 1, window_size)
    strides = arr.strides + (arr.strides[-1],)
    strided = as_strided(arr, shape=shape, strides=strides)

    if window_step > 1:
        strided = strided[..., ::window_step, :]

    orig = list(range(len(strided.shape)))
    trans = list(range(len(strided.shape)))
    trans[-2] = orig[-1]
    trans[-1] = orig[-2]
    trans = trans[::-1]
    strided = strided.transpose(trans)
    return strided


def unwindow(arr, window_size, window_step=1, axis=0):
    # undo windows by broadcast
    if axis != 0:
        raise ValueError("axis != 0 currently unsupported")
    shp = arr.shape
    unwindowed = np.tile(arr[:, None, ...], (1, window_step, 1, 1))
    unwindowed = unwindowed.reshape(shp[0] * window_step, *shp[1:])
    return unwindowed.mean(axis=1)


def xcorr_offset(x1, x2):
    """
    Under MSR-LA License

    Based on MATLAB implementation from Spectrogram Inversion Toolbox

    References
    ----------
    D. Griffin and J. Lim. Signal estimation from modified
    short-time Fourier transform. IEEE Trans. Acoust. Speech
    Signal Process., 32(2):236-243, 1984.

    Malcolm Slaney, Daniel Naar and Richard F. Lyon. Auditory
    Model Inversion for Sound Separation. Proc. IEEE-ICASSP,
    Adelaide, 1994, II.77-80.

    Xinglei Zhu, G. Beauregard, L. Wyse. Real-Time Signal
    Estimation from Modified Short-Time Fourier Transform
    Magnitude Spectra. IEEE Transactions on Audio Speech and
    Language Processing, 08/2007.
    """
    x1 = x1 - x1.mean()
    x2 = x2 - x2.mean()
    frame_size = len(x2)
    half = frame_size // 2
    corrs = np.convolve(x1.astype('float32'), x2[::-1].astype('float32'))
    corrs[:half] = -1E30
    corrs[-half:] = -1E30
    offset = corrs.argmax() - len(x1)
    return offset


def invert_spectrogram(X_s, step, calculate_offset=True, set_zero_phase=True):
    """
    Under MSR-LA License

    Based on MATLAB implementation from Spectrogram Inversion Toolbox

    References
    ----------
    D. Griffin and J. Lim. Signal estimation from modified
    short-time Fourier transform. IEEE Trans. Acoust. Speech
    Signal Process., 32(2):236-243, 1984.

    Malcolm Slaney, Daniel Naar and Richard F. Lyon. Auditory
    Model Inversion for Sound Separation. Proc. IEEE-ICASSP,
    Adelaide, 1994, II.77-80.

    Xinglei Zhu, G. Beauregard, L. Wyse. Real-Time Signal
    Estimation from Modified Short-Time Fourier Transform
    Magnitude Spectra. IEEE Transactions on Audio Speech and
    Language Processing, 08/2007.
    """
    size = int(X_s.shape[1] // 2)
    wave = np.zeros((X_s.shape[0] * step + size))
    # Getting overflow warnings with 32 bit...
    wave = wave.astype('float64')
    total_windowing_sum = np.zeros((X_s.shape[0] * step + size))
    win = 0.54 - .46 * np.cos(2 * np.pi * np.arange(size) / (size - 1))

    est_start = int(size // 2) - 1
    est_end = est_start + size
    for i in range(X_s.shape[0]):
        wave_start = int(step * i)
        wave_end = wave_start + size
        if set_zero_phase:
            spectral_slice = X_s[i].real + 0j
        else:
            # already complex
            spectral_slice = X_s[i]

        # Don't need fftshift due to different impl.
        wave_est = np.real(np.fft.ifft(spectral_slice))[::-1]
        if calculate_offset and i > 0:
            offset_size = size - step
            if offset_size <= 0:
                print("WARNING: Large step size >50\% detected! "
                      "This code works best with high overlap - try "
                      "with 75% or greater")
                offset_size = step
            offset = xcorr_offset(wave[wave_start:wave_start + offset_size],
                                  wave_est[est_start:est_start + offset_size])
        else:
            offset = 0
        wave[wave_start:wave_end] += win * wave_est[
            est_start - offset:est_end - offset]
        total_windowing_sum[wave_start:wave_end] += win
    wave = np.real(wave) / (total_windowing_sum + 1E-6)
    return wave


def iterate_invert_spectrogram(X_s, fftsize, step, n_iter=10, verbose=False,
                               complex_input=False):
    """
    Under MSR-LA License

    Based on MATLAB implementation from Spectrogram Inversion Toolbox

    References
    ----------
    D. Griffin and J. Lim. Signal estimation from modified
    short-time Fourier transform. IEEE Trans. Acoust. Speech
    Signal Process., 32(2):236-243, 1984.

    Malcolm Slaney, Daniel Naar and Richard F. Lyon. Auditory
    Model Inversion for Sound Separation. Proc. IEEE-ICASSP,
    Adelaide, 1994, II.77-80.

    Xinglei Zhu, G. Beauregard, L. Wyse. Real-Time Signal
    Estimation from Modified Short-Time Fourier Transform
    Magnitude Spectra. IEEE Transactions on Audio Speech and
    Language Processing, 08/2007.
    """
    reg = np.max(X_s) / 1E8
    X_best = copy.deepcopy(X_s)
    try:
        for i in range(n_iter):
            if verbose:
                print("Runnning iter %i" % i)
            if i == 0 and not complex_input:
                X_t = invert_spectrogram(X_best, step, calculate_offset=True,
                                        set_zero_phase=True)
            else:
                # Calculate offset was False in the MATLAB version
                # but in mine it massively improves the result
                # Possible bug in my impl?
                X_t = invert_spectrogram(X_best, step, calculate_offset=True,
                                        set_zero_phase=False)
            est = stft(X_t, fftsize=fftsize, step=step, compute_onesided=False)
            phase = est / np.maximum(reg, np.abs(est))
            phase = phase[:len(X_s)]
            X_s = X_s[:len(phase)]
            X_best = X_s * phase
    except ValueError:
        raise ValueError("The iterate_invert_spectrogram algorithm requires"
                         " stft(..., compute_onesided=False),",
                         " be sure you have calculated stft with this argument")
    X_t = invert_spectrogram(X_best, step, calculate_offset=True,
                             set_zero_phase=False)
    return np.real(X_t)


def harvest_get_downsampled_signal(x, fs, target_fs):
    decimation_ratio = np.round(fs / target_fs)
    offset = np.ceil(140. / decimation_ratio) * decimation_ratio
    start_pad = x[0] * np.ones(int(offset), dtype=np.float32)
    end_pad = x[-1] * np.ones(int(offset), dtype=np.float32)
    x = np.concatenate((start_pad, x, end_pad), axis=0)

    if fs < target_fs:
        raise ValueError("CASE NOT HANDLED IN harvest_get_downsampled_signal")
    else:
        try:
            y0 = sg.decimate(x, int(decimation_ratio), 3, zero_phase=True)
        except:
            y0 = sg.decimate(x, int(decimation_ratio), 3)
        actual_fs = fs / decimation_ratio
        y = y0[int(offset / decimation_ratio):-int(offset / decimation_ratio)]
    y = y - np.mean(y)
    return y, actual_fs


def harvest_get_raw_f0_candidates(number_of_frames, boundary_f0_list,
      y_length, temporal_positions, actual_fs, y_spectrum, f0_floor,
      f0_ceil):
    raw_f0_candidates = np.zeros((len(boundary_f0_list), number_of_frames), dtype=np.float32)
    for i in range(len(boundary_f0_list)):
        raw_f0_candidates[i, :] = harvest_get_f0_candidate_from_raw_event(
                boundary_f0_list[i], actual_fs, y_spectrum, y_length,
                temporal_positions, f0_floor, f0_ceil)
    return raw_f0_candidates


def harvest_nuttall(N):
    t = np.arange(0, N) * 2 * np.pi / (N - 1)
    coefs = np.array([0.355768, -0.487396, 0.144232, -0.012604])
    window = np.cos(t[:, None].dot(np.array([0., 1., 2., 3.])[None])).dot( coefs[:, None])
    # 1D window...
    return window.ravel()


def harvest_get_f0_candidate_from_raw_event(boundary_f0,
        fs, y_spectrum, y_length, temporal_positions, f0_floor,
        f0_ceil):
    filter_length_half = int(np.round(fs / boundary_f0 * 2))
    band_pass_filter_base = harvest_nuttall(filter_length_half * 2 + 1)
    shifter = np.cos(2 * np.pi * boundary_f0 * np.arange(-filter_length_half, filter_length_half + 1) / float(fs))
    band_pass_filter = band_pass_filter_base * shifter

    index_bias = filter_length_half
    # possible numerical issues if 32 bit
    spectrum_low_pass_filter = np.fft.fft(band_pass_filter.astype("float64"), len(y_spectrum))
    filtered_signal = np.real(np.fft.ifft(spectrum_low_pass_filter * y_spectrum))
    index_bias = filter_length_half + 1
    filtered_signal = filtered_signal[index_bias + np.arange(y_length).astype("int32")]
    negative_zero_cross = harvest_zero_crossing_engine(filtered_signal, fs)
    positive_zero_cross = harvest_zero_crossing_engine(-filtered_signal, fs)
    d_filtered_signal = filtered_signal[1:] - filtered_signal[:-1]
    peak = harvest_zero_crossing_engine(d_filtered_signal, fs)
    dip = harvest_zero_crossing_engine(-d_filtered_signal, fs)
    f0_candidate = harvest_get_f0_candidate_contour(negative_zero_cross,
            positive_zero_cross, peak, dip, temporal_positions)
    f0_candidate[f0_candidate > (boundary_f0 * 1.1)] = 0.
    f0_candidate[f0_candidate < (boundary_f0 * .9)] = 0.
    f0_candidate[f0_candidate > f0_ceil] = 0.
    f0_candidate[f0_candidate < f0_floor] = 0.
    return f0_candidate


def harvest_get_f0_candidate_contour(negative_zero_cross_tup,
        positive_zero_cross_tup, peak_tup, dip_tup, temporal_positions):
    # 0 is inteval locations
    # 1 is interval based f0
    usable_channel = max(0, len(negative_zero_cross_tup[0]) - 2)
    usable_channel *= max(0, len(positive_zero_cross_tup[0]) - 2)
    usable_channel *= max(0, len(peak_tup[0]) - 2)
    usable_channel *= max(0, len(dip_tup[0]) - 2)
    if usable_channel > 0:
        interpolated_f0_list = np.zeros((4, len(temporal_positions)))
        nz = interp1d(negative_zero_cross_tup[0], negative_zero_cross_tup[1],
                 kind="linear", bounds_error=False, fill_value="extrapolate")
        pz = interp1d(positive_zero_cross_tup[0], positive_zero_cross_tup[1],
                 kind="linear", bounds_error=False, fill_value="extrapolate")
        pkz = interp1d(peak_tup[0], peak_tup[1],
                  kind="linear", bounds_error=False, fill_value="extrapolate")
        dz = interp1d(dip_tup[0], dip_tup[1],
                  kind="linear", bounds_error=False, fill_value="extrapolate")
        interpolated_f0_list[0, :] = nz(temporal_positions)
        interpolated_f0_list[1, :] = pz(temporal_positions)
        interpolated_f0_list[2, :] = pkz(temporal_positions)
        interpolated_f0_list[3, :] = dz(temporal_positions)
        f0_candidate = np.mean(interpolated_f0_list, axis=0)
    else:
        f0_candidate = temporal_positions * 0
    return f0_candidate


def harvest_zero_crossing_engine(x, fs, debug=False):
    # negative zero crossing, going from positive to negative
    x_shift = x.copy()
    x_shift[:-1] = x_shift[1:]
    x_shift[-1] = x[-1]
    # +1 here to avoid edge case at 0
    points = np.arange(len(x)) + 1
    negative_going_points = points * ((x_shift * x < 0) * (x_shift < x))
    edge_list = negative_going_points[negative_going_points > 0]
    # -1 to correct index
    fine_edge_list = edge_list - x[edge_list - 1] / (x[edge_list] - x[edge_list - 1]).astype("float32")
    interval_locations = (fine_edge_list[:-1] + fine_edge_list[1:]) / float(2) / fs
    interval_based_f0 = float(fs) / (fine_edge_list[1:] - fine_edge_list[:-1])
    return interval_locations, interval_based_f0


def harvest_detect_official_f0_candidates(raw_f0_candidates):
    number_of_channels, number_of_frames = raw_f0_candidates.shape
    f0_candidates = np.zeros((int(np.round(number_of_channels / 10.)), number_of_frames))
    number_of_candidates = 0
    threshold = 10
    for i in range(number_of_frames):
        tmp = raw_f0_candidates[:, i].copy()
        tmp[tmp > 0] = 1.
        tmp[0] = 0
        tmp[-1] = 0
        tmp = tmp[1:] - tmp[:-1]
        st = np.where(tmp == 1)[0]
        ed = np.where(tmp == -1)[0]
        count = 0
        for j in range(len(st)):
            dif = ed[j] - st[j]
            if dif >= threshold:
                tmp_f0 = raw_f0_candidates[st[j] + 1: ed[j] + 1, i]
                f0_candidates[count, i] = np.mean(tmp_f0)
                count = count + 1
        number_of_candidates = max(number_of_candidates, count)
    return f0_candidates, number_of_candidates


def harvest_overlap_f0_candidates(f0_candidates, max_number_of_f0_candidates):
    n = 3 # this is the optimized parameter... apparently
    number_of_candidates = n * 2 + 1
    new_f0_candidates = f0_candidates[number_of_candidates, :].copy()
    new_f0_candidates = new_f0_candidates[None]
    # hack to bypass magic matlab-isms of allocating when indexing OOB
    new_f0_candidates = np.vstack([new_f0_candidates] + (new_f0_candidates.shape[-1] - 1) * [np.zeros_like(new_f0_candidates)])
    # this indexing is megagross, possible source for bugs!
    all_nonzero = []
    for i in range(number_of_candidates):
        st = max(-(i - n), 0)
        ed = min(-(i - n), 0)
        f1_b = np.arange(max_number_of_f0_candidates).astype("int32")
        f1 = f1_b + int(i * max_number_of_f0_candidates)
        all_nonzero = list(set(all_nonzero + list(f1)))
        f2 = None if ed == 0 else ed
        f3 = -ed
        f4 = None if st == 0 else -st
        new_f0_candidates[f1, st:f2] = f0_candidates[f1_b, f3:f4]
    new_f0_candidates = new_f0_candidates[all_nonzero, :]
    return new_f0_candidates


def harvest_refine_candidates(x, fs, temporal_positions, f0_candidates,
        f0_floor, f0_ceil):
    new_f0_candidates = f0_candidates.copy()
    f0_scores = f0_candidates * 0.
    for i in range(len(temporal_positions)):
        for j in range(len(f0_candidates)):
            tmp_f0 = f0_candidates[j, i]
            if tmp_f0 == 0:
                continue
            res = harvest_get_refined_f0(x, fs, temporal_positions[i],
                    tmp_f0, f0_floor, f0_ceil)
            new_f0_candidates[j, i] = res[0]
            f0_scores[j, i] = res[1]
    return new_f0_candidates, f0_scores


def harvest_get_refined_f0(x, fs, current_time, current_f0, f0_floor,
        f0_ceil):
    half_window_length = np.ceil(3. * fs / current_f0 / 2.)
    window_length_in_time = (2. * half_window_length + 1) / float(fs)
    base_time = np.arange(-half_window_length, half_window_length + 1) / float(fs)
    fft_size = int(2 ** np.ceil(np.log2((half_window_length * 2 + 1)) + 1))
    frequency_axis = np.arange(fft_size) / fft_size * float(fs)

    base_index = np.round((current_time + base_time) * fs + 0.001)
    index_time = (base_index - 1) / float(fs)
    window_time = index_time - current_time
    part1 = np.cos(2 * np.pi * window_time / window_length_in_time)
    part2 = np.cos(4 * np.pi * window_time / window_length_in_time)
    main_window = 0.42 + 0.5 * part1 + 0.08 * part2
    ext = np.zeros((len(main_window) + 2))
    ext[1:-1] = main_window
    diff_window = -((ext[1:-1] - ext[:-2]) + (ext[2:] - ext[1:-1])) / float(2)
    safe_index = np.maximum(1, np.minimum(len(x), base_index)).astype("int32") - 1
    spectrum = np.fft.fft(x[safe_index] * main_window, fft_size)
    diff_spectrum = np.fft.fft(x[safe_index] * diff_window, fft_size)
    numerator_i = np.real(spectrum) * np.imag(diff_spectrum) - np.imag(spectrum) * np.real(diff_spectrum)
    power_spectrum = np.abs(spectrum) ** 2
    instantaneous_frequency = frequency_axis + numerator_i / power_spectrum * float(fs) / 2. / np.pi

    number_of_harmonics = int(min(np.floor(float(fs) / 2. / current_f0), 6.))
    harmonics_index = np.arange(number_of_harmonics) + 1
    index_list = np.round(current_f0 * fft_size / fs * harmonics_index).astype("int32")
    instantaneous_frequency_list = instantaneous_frequency[index_list]
    amplitude_list = np.sqrt(power_spectrum[index_list])
    refined_f0 = np.sum(amplitude_list * instantaneous_frequency_list)
    refined_f0 /= np.sum(amplitude_list * harmonics_index.astype("float32"))

    variation = np.abs(((instantaneous_frequency_list / harmonics_index.astype("float32")) - current_f0) / float(current_f0))
    refined_score = 1. / (0.000000000001 + np.mean(variation))

    if (refined_f0 < f0_floor) or (refined_f0 > f0_ceil) or (refined_score < 2.5):
        refined_f0 = 0.
        redined_score = 0.
    return refined_f0, refined_score


def harvest_select_best_f0(reference_f0, f0_candidates, allowed_range):
    best_f0 = 0
    best_error = allowed_range

    for i in range(len(f0_candidates)):
        tmp = np.abs(reference_f0 - f0_candidates[i]) / reference_f0
        if tmp > best_error:
            continue
        best_f0 = f0_candidates[i]
        best_error = tmp
    return best_f0, best_error


def harvest_remove_unreliable_candidates(f0_candidates, f0_scores):
    new_f0_candidates = f0_candidates.copy()
    new_f0_scores = f0_scores.copy()
    threshold = 0.05
    f0_length = f0_candidates.shape[1]
    number_of_candidates = len(f0_candidates)

    for i in range(1, f0_length - 1):
        for j in range(number_of_candidates):
            reference_f0 = f0_candidates[j, i]
            if reference_f0 == 0:
                continue
            _, min_error1 = harvest_select_best_f0(reference_f0, f0_candidates[:, i + 1], 1)
            _, min_error2 = harvest_select_best_f0(reference_f0, f0_candidates[:, i - 1], 1)
            min_error = min([min_error1, min_error2])
            if min_error > threshold:
                new_f0_candidates[j, i] = 0
                new_f0_scores[j, i] = 0
    return new_f0_candidates, new_f0_scores


def harvest_search_f0_base(f0_candidates, f0_scores):
    f0_base = f0_candidates[0, :] * 0.
    for i in range(len(f0_base)):
        max_index = np.argmax(f0_scores[:, i])
        f0_base[i] = f0_candidates[max_index, i]
    return f0_base


def harvest_fix_step_1(f0_base, allowed_range):
    # Step 1: Rapid change of f0 contour is replaced by 0
    f0_step1 = f0_base.copy()
    f0_step1[0] = 0.
    f0_step1[1] = 0.

    for i in range(2, len(f0_base)):
        if f0_base[i] == 0:
            continue
        reference_f0 = f0_base[i - 1] * 2 - f0_base[i - 2]
        c1 = np.abs((f0_base[i] - reference_f0) / reference_f0) > allowed_range
        c2 = np.abs((f0_base[i] - f0_base[i - 1]) / f0_base[i - 1]) > allowed_range
        if c1 and c2:
            f0_step1[i] = 0.
    return f0_step1


def harvest_fix_step_2(f0_step1, voice_range_minimum):
    f0_step2 = f0_step1.copy()
    boundary_list = harvest_get_boundary_list(f0_step1)

    for i in range(1, int(len(boundary_list) / 2.) + 1):
        distance = boundary_list[(2 * i) - 1] - boundary_list[(2 * i) - 2]
        if distance < voice_range_minimum:
            # need one more due to range not including last index
            lb = boundary_list[(2 * i) - 2]
            ub = boundary_list[(2 * i) - 1] + 1
            f0_step2[lb:ub] = 0.
    return f0_step2


def harvest_fix_step_3(f0_step2, f0_candidates, allowed_range, f0_scores):
    f0_step3 = f0_step2.copy()
    boundary_list = harvest_get_boundary_list(f0_step2)
    multichannel_f0 = harvest_get_multichannel_f0(f0_step2, boundary_list)
    rrange = np.zeros((int(len(boundary_list) / 2), 2))
    threshold1 = 100
    threshold2 = 2200
    count = 0
    for i in range(1, int(len(boundary_list) / 2) + 1):
        extended_f0, tmp_range_1 = harvest_extend_f0(multichannel_f0[i - 1, :],
                boundary_list[(2 * i) - 1],
                min([len(f0_step2) - 1, boundary_list[(2 * i) - 1] + threshold1]),
                1, f0_candidates, allowed_range)
        tmp_f0_sequence, tmp_range_0 = harvest_extend_f0(extended_f0,
                boundary_list[(2 * i) - 2],
                max([2, boundary_list[(2 * i) - 2] - threshold1]), -1,
                f0_candidates, allowed_range)

        mean_f0 = np.mean(tmp_f0_sequence[tmp_range_0 : tmp_range_1 + 1])
        if threshold2 / mean_f0 < (tmp_range_1 - tmp_range_0):
            multichannel_f0[count, :] = tmp_f0_sequence
            rrange[count, :] = np.array([tmp_range_0, tmp_range_1])
            count = count + 1
    if count > 0:
        multichannel_f0 = multichannel_f0[:count, :]
        rrange = rrange[:count, :]
        f0_step3 = harvest_merge_f0(multichannel_f0, rrange, f0_candidates,
                f0_scores)
    return f0_step3


def harvest_merge_f0(multichannel_f0, rrange, f0_candidates, f0_scores):
    number_of_channels = len(multichannel_f0)
    sorted_order = np.argsort(rrange[:, 0])
    f0 = multichannel_f0[sorted_order[0], :]
    for i in range(1, number_of_channels):
        if rrange[sorted_order[i], 0] - rrange[sorted_order[0], 1] > 0:
            # no overlapping
            f0[rrange[sorted_order[i], 0]:rrange[sorted_order[i], 1]] = multichannel_f0[sorted_order[i], rrange[sorted_order[i], 0]:rrange[sorted_order[i], 1]]
            cp = rrange.copy()
            rrange[sorted_order[0], 0] = cp[sorted_order[i], 0]
            rrange[sorted_order[0], 1] = cp[sorted_order[i], 1]
        else:
            cp = rrange.copy()
            res = harvest_merge_f0_sub(f0, cp[sorted_order[0], 0],
                    cp[sorted_order[0], 1],
                    multichannel_f0[sorted_order[i], :],
                    cp[sorted_order[i], 0],
                    cp[sorted_order[i], 1], f0_candidates, f0_scores)
            f0 = res[0]
            rrange[sorted_order[0], 1] = res[1]
    return f0


def harvest_merge_f0_sub(f0_1, st1, ed1, f0_2, st2, ed2, f0_candidates,
        f0_scores):
    merged_f0 = f0_1
    if (st1 <= st2) and (ed1 >= ed2):
        new_ed = ed1
        return merged_f0, new_ed
    new_ed = ed2

    score1 = 0.
    score2 = 0.
    for i in range(int(st2), int(ed1) + 1):
        score1 = score1 + harvest_serach_score(f0_1[i], f0_candidates[:, i], f0_scores[:, i])
        score2 = score2 + harvest_serach_score(f0_2[i], f0_candidates[:, i], f0_scores[:, i])
    if score1 > score2:
        merged_f0[int(ed1):int(ed2) + 1] = f0_2[int(ed1):int(ed2) + 1]
    else:
        merged_f0[int(st2):int(ed2) + 1] = f0_2[int(st2):int(ed2) + 1]
    return merged_f0, new_ed


def harvest_serach_score(f0, f0_candidates, f0_scores):
    score = 0
    for i in range(len(f0_candidates)):
        if (f0 == f0_candidates[i]) and (score < f0_scores[i]):
            score = f0_scores[i]
    return score


def harvest_extend_f0(f0, origin, last_point, shift, f0_candidates,
        allowed_range):
    threshold = 4
    extended_f0 = f0.copy()
    tmp_f0 = extended_f0[origin]
    shifted_origin = origin
    count = 0
    for i in np.arange(origin, last_point + shift, shift):
        bf0, bs = harvest_select_best_f0(tmp_f0,
                f0_candidates[:, i + shift], allowed_range)
        extended_f0[i + shift] = bf0
        if extended_f0[i + shift] != 0:
            tmp_f0 = extended_f0[i + shift]
            count = 0
            shifted_origin = i + shift
        else:
            count = count + 1
        if count == threshold:
            break
    return extended_f0, shifted_origin


def harvest_get_multichannel_f0(f0, boundary_list):
    multichannel_f0 = np.zeros((int(len(boundary_list) / 2), len(f0)))
    for i in range(1, int(len(boundary_list) / 2) + 1):
        sl = boundary_list[(2 * i) - 2]
        el = boundary_list[(2 * i) - 1] + 1
        multichannel_f0[i - 1, sl:el] = f0[sl:el]
    return multichannel_f0


def harvest_get_boundary_list(f0):
    vuv = f0.copy()
    vuv[vuv != 0] = 1.
    vuv[0] = 0
    vuv[-1] = 0
    diff_vuv = vuv[1:] - vuv[:-1]
    boundary_list = np.where(diff_vuv != 0)[0]
    boundary_list[::2] = boundary_list[::2] + 1
    return boundary_list


def harvest_fix_step_4(f0_step3, threshold):
    f0_step4 = f0_step3.copy()
    boundary_list = harvest_get_boundary_list(f0_step3)

    for i in range(1, int(len(boundary_list) / 2.)):
        distance = boundary_list[(2 * i)] - boundary_list[(2 * i) - 1] - 1
        if distance >= threshold:
            continue
        boundary0 = f0_step3[boundary_list[(2 * i) - 1]] + 1
        boundary1 = f0_step3[boundary_list[(2 * i)]] - 1
        coefficient = (boundary1 - boundary0) / float((distance + 1))
        count = 1
        st = boundary_list[(2 * i) - 1] + 1
        ed = boundary_list[(2 * i)]
        for j in range(st, ed):
            f0_step4[j] = boundary0 + coefficient * count
            count = count + 1
    return f0_step4


def harvest_fix_f0_contour(f0_candidates, f0_scores):
    f0_base = harvest_search_f0_base(f0_candidates, f0_scores)
    f0_step1 = harvest_fix_step_1(f0_base, 0.008) # optimized?
    f0_step2 = harvest_fix_step_2(f0_step1, 6) # optimized?
    f0_step3 = harvest_fix_step_3(f0_step2, f0_candidates, 0.18, f0_scores) # optimized?
    f0 = harvest_fix_step_4(f0_step3, 9) # optimized
    vuv = f0.copy()
    vuv[vuv != 0] = 1.
    return f0, vuv


def harvest_filter_f0_contour(f0, st, ed, b, a):
    smoothed_f0 = f0.copy()
    smoothed_f0[:st] = smoothed_f0[st]
    smoothed_f0[ed + 1:] = smoothed_f0[ed]
    aaa = sg.lfilter(b, a, smoothed_f0)
    bbb = sg.lfilter(b, a, aaa[::-1])
    smoothed_f0 = bbb[::-1].copy()
    smoothed_f0[:st] = 0.
    smoothed_f0[ed + 1:] = 0.
    return smoothed_f0


def harvest_smooth_f0_contour(f0):
    b = np.array([0.0078202080334971724, 0.015640416066994345, 0.0078202080334971724])
    a = np.array([1.0, -1.7347257688092754, 0.76600660094326412])
    smoothed_f0 = np.concatenate([np.zeros(300,), f0, np.zeros(300,)])
    boundary_list = harvest_get_boundary_list(smoothed_f0)
    multichannel_f0 = harvest_get_multichannel_f0(smoothed_f0, boundary_list)
    for i in range(1, int(len(boundary_list) / 2) + 1):
        tmp_f0_contour = harvest_filter_f0_contour(multichannel_f0[i - 1, :],
                boundary_list[(2 * i) - 2], boundary_list[(2 * i) - 1], b, a)
        st = boundary_list[(2 * i) - 2]
        ed = boundary_list[(2 * i) - 1] + 1
        smoothed_f0[st:ed] = tmp_f0_contour[st:ed]
    smoothed_f0 = smoothed_f0[300:-300]
    return smoothed_f0


def _world_get_temporal_positions(x_len, fs):
    frame_period = 5
    basic_frame_period = 1
    basic_temporal_positions = np.arange(0, x_len / float(fs), basic_frame_period / float(1000))
    temporal_positions = np.arange(0,
            x_len / float(fs),
            frame_period / float(1000))
    return basic_temporal_positions, temporal_positions


def harvest(x, fs):
    f0_floor = 71
    f0_ceil = 800
    target_fs = 8000
    channels_in_octave = 40.
    basic_temporal_positions, temporal_positions = _world_get_temporal_positions(len(x), fs)
    adjusted_f0_floor = f0_floor * 0.9
    adjusted_f0_ceil = f0_ceil * 1.1
    boundary_f0_list = np.arange(1, np.ceil(np.log2(adjusted_f0_ceil / adjusted_f0_floor) * channels_in_octave) + 1) / float(channels_in_octave)
    boundary_f0_list = adjusted_f0_floor * 2.0 ** boundary_f0_list
    y, actual_fs = harvest_get_downsampled_signal(x, fs, target_fs)
    fft_size = 2. ** np.ceil(np.log2(len(y) + np.round(fs / f0_floor * 4) + 1))
    y_spectrum = np.fft.fft(y, int(fft_size))
    raw_f0_candidates = harvest_get_raw_f0_candidates(
        len(basic_temporal_positions),
        boundary_f0_list, len(y), basic_temporal_positions, actual_fs,
        y_spectrum, f0_floor, f0_ceil)

    f0_candidates, number_of_candidates = harvest_detect_official_f0_candidates(raw_f0_candidates)
    f0_candidates = harvest_overlap_f0_candidates(f0_candidates, number_of_candidates)
    f0_candidates, f0_scores = harvest_refine_candidates(y, actual_fs,
            basic_temporal_positions, f0_candidates, f0_floor, f0_ceil)

    f0_candidates, f0_scores = harvest_remove_unreliable_candidates(f0_candidates, f0_scores)

    connected_f0, vuv = harvest_fix_f0_contour(f0_candidates, f0_scores)
    smoothed_f0 = harvest_smooth_f0_contour(connected_f0)
    idx = np.minimum(len(smoothed_f0) - 1, np.round(temporal_positions * 1000)).astype("int32")
    f0 = smoothed_f0[idx]
    vuv = vuv[idx]
    f0_candidates = f0_candidates
    return temporal_positions, f0, vuv, f0_candidates


def cheaptrick_get_windowed_waveform(x, fs, current_f0, current_position):
    half_window_length = np.round(1.5 * fs / float(current_f0))
    base_index = np.arange(-half_window_length, half_window_length + 1)
    index = np.round(current_position * fs + 0.001) + base_index + 1
    safe_index = np.minimum(len(x), np.maximum(1, np.round(index))).astype("int32")
    safe_index = safe_index - 1
    segment = x[safe_index]
    time_axis = base_index / float(fs) / 1.5
    window1 = 0.5 * np.cos(np.pi * time_axis * float(current_f0)) + 0.5
    window1 = window1 / np.sqrt(np.sum(window1 ** 2))
    waveform = segment * window1 - window1 * np.mean(segment * window1) / np.mean(window1)
    return waveform


def cheaptrick_get_power_spectrum(waveform, fs, fft_size, f0):
    power_spectrum = np.abs(np.fft.fft(waveform, fft_size)) ** 2
    frequency_axis = np.arange(fft_size) / float(fft_size) * float(fs)
    ind = frequency_axis < (f0 + fs / fft_size)
    low_frequency_axis = frequency_axis[ind]
    low_frequency_replica = interp1d(f0 - low_frequency_axis,
            power_spectrum[ind], kind="linear",
            fill_value="extrapolate")(low_frequency_axis)
    p1 = low_frequency_replica[(frequency_axis < f0)[:len(low_frequency_replica)]]
    p2 = power_spectrum[(frequency_axis < f0)[:len(power_spectrum)]]
    power_spectrum[frequency_axis < f0] = p1 + p2
    lb1 = int(fft_size / 2) + 1
    lb2 = 1
    ub2 = int(fft_size / 2)
    power_spectrum[lb1:] = power_spectrum[lb2:ub2][::-1]
    return power_spectrum


def cheaptrick_linear_smoothing(power_spectrum, f0, fs, fft_size):
    double_frequency_axis = np.arange(2 * fft_size) / float(fft_size ) * fs - fs
    double_spectrum = np.concatenate([power_spectrum, power_spectrum])

    double_segment = np.cumsum(double_spectrum * (fs / float(fft_size)))
    center_frequency = np.arange(int(fft_size / 2) + 1) / float(fft_size ) * fs
    low_levels = cheaptrick_interp1h(double_frequency_axis + fs / float(fft_size) / 2.,
            double_segment, center_frequency - f0 / 3.)
    high_levels = cheaptrick_interp1h(double_frequency_axis + fs / float(fft_size) / 2.,
            double_segment, center_frequency + f0 / 3.)
    smoothed_spectrum = (high_levels - low_levels) * 1.5 / f0
    return smoothed_spectrum


def cheaptrick_interp1h(x, y, xi):
    delta_x = float(x[1] - x[0])
    xi = np.maximum(x[0], np.minimum(x[-1], xi))
    xi_base = (np.floor((xi - x[0]) / delta_x)).astype("int32")
    xi_fraction = (xi - x[0]) / delta_x - xi_base
    delta_y = np.zeros_like(y)
    delta_y[:-1] = y[1:] - y[:-1]
    yi = y[xi_base] + delta_y[xi_base] * xi_fraction
    return yi


def cheaptrick_smoothing_with_recovery(smoothed_spectrum, f0, fs, fft_size, q1):
    quefrency_axis = np.arange(fft_size) / float(fs)
    # 0 is NaN
    smoothing_lifter = np.sin(np.pi * f0 * quefrency_axis) / (np.pi * f0 * quefrency_axis)
    p = smoothing_lifter[1:int(fft_size / 2)][::-1].copy()
    smoothing_lifter[int(fft_size / 2) + 1:] = p
    smoothing_lifter[0] = 1.
    compensation_lifter = (1 - 2. * q1) + 2. * q1 * np.cos(2 * np.pi * quefrency_axis * f0)
    p = compensation_lifter[1:int(fft_size / 2)][::-1].copy()
    compensation_lifter[int(fft_size / 2) + 1:] = p
    tandem_cepstrum = np.fft.fft(np.log(smoothed_spectrum))
    tmp_spectral_envelope = np.exp(np.real(np.fft.ifft(tandem_cepstrum * smoothing_lifter * compensation_lifter)))
    spectral_envelope = tmp_spectral_envelope[:int(fft_size / 2) + 1]
    return spectral_envelope


def cheaptrick_estimate_one_slice(x, fs, current_f0,
    current_position, fft_size, q1):
    waveform = cheaptrick_get_windowed_waveform(x, fs, current_f0,
        current_position)
    power_spectrum = cheaptrick_get_power_spectrum(waveform, fs, fft_size,
            current_f0)
    smoothed_spectrum = cheaptrick_linear_smoothing(power_spectrum, current_f0,
            fs, fft_size)
    comb_spectrum = np.concatenate([smoothed_spectrum, smoothed_spectrum[1:-1][::-1]])
    spectral_envelope = cheaptrick_smoothing_with_recovery(comb_spectrum,
            current_f0, fs, fft_size, q1)
    return spectral_envelope


def cheaptrick(x, fs, temporal_positions, f0_sequence,
        vuv, fftlen="auto", q1=-0.15):
    f0_sequence = f0_sequence.copy()
    f0_low_limit = 71
    default_f0 = 500
    if fftlen == "auto":
        fftlen = int(2 ** np.ceil(np.log2(3. * float(fs) / f0_low_limit + 1)))
    #raise ValueError("Only fftlen auto currently supported")
    fft_size = fftlen
    f0_low_limit = fs * 3.0 / (fft_size - 3.0)
    f0_sequence[vuv == 0] = default_f0
    spectrogram = np.zeros((int(fft_size / 2.) + 1, len(f0_sequence)))
    for i in range(len(f0_sequence)):
        if f0_sequence[i] < f0_low_limit:
            f0_sequence[i] = default_d0
        spectrogram[:, i] = cheaptrick_estimate_one_slice(x, fs, f0_sequence[i],
                temporal_positions[i], fft_size, q1)
    return temporal_positions, spectrogram.T, fs


def d4c_love_train(x, fs, current_f0, current_position, threshold):
    vuv = 0
    if current_f0 == 0:
        return vuv
    lowest_f0 = 40
    current_f0 = max([current_f0, lowest_f0])
    fft_size = int(2 ** np.ceil(np.log2(3. * fs / lowest_f0 + 1)))
    boundary0 = int(np.ceil(100 / (float(fs) / fft_size)))
    boundary1 = int(np.ceil(4000 / (float(fs) / fft_size)))
    boundary2 = int(np.ceil(7900 / (float(fs) / fft_size)))

    waveform = d4c_get_windowed_waveform(x, fs, current_f0, current_position,
            1.5, 2)
    power_spectrum = np.abs(np.fft.fft(waveform, int(fft_size)) ** 2)
    power_spectrum[0:boundary0 + 1] = 0.
    cumulative_spectrum = np.cumsum(power_spectrum)
    if (cumulative_spectrum[boundary1] / cumulative_spectrum[boundary2]) > threshold:
        vuv = 1
    return vuv


def d4c_get_windowed_waveform(x, fs, current_f0, current_position, half_length,
        window_type):
    half_window_length = int(np.round(half_length * fs / current_f0))
    base_index = np.arange(-half_window_length, half_window_length + 1)
    index = np.round(current_position * fs + 0.001) + base_index + 1
    safe_index = np.minimum(len(x), np.maximum(1, np.round(index))).astype("int32") - 1

    segment = x[safe_index]
    time_axis = base_index / float(fs) / float(half_length)
    if window_type == 1:
        window1 = 0.5 * np.cos(np.pi * time_axis * current_f0) + 0.5
    elif window_type == 2:
        window1 = 0.08 * np.cos(np.pi * time_axis * current_f0 * 2)
        window1 += 0.5 * np.cos(np.pi * time_axis * current_f0) + 0.42
    else:
        raise ValueError("Unknown window type")
    waveform = segment * window1 - window1 * np.mean(segment * window1) / np.mean(window1)
    return waveform


def d4c_get_static_centroid(x, fs, current_f0, current_position, fft_size):
    waveform1 = d4c_get_windowed_waveform(x, fs, current_f0,
        current_position + 1. / current_f0 / 4., 2, 2)
    waveform2 = d4c_get_windowed_waveform(x, fs, current_f0,
        current_position - 1. / current_f0 / 4., 2, 2)
    centroid1 = d4c_get_centroid(waveform1, fft_size)
    centroid2 = d4c_get_centroid(waveform2, fft_size)
    centroid = d4c_dc_correction(centroid1 + centroid2, fs, fft_size,
            current_f0)
    return centroid


def d4c_get_centroid(x, fft_size):
    fft_size = int(fft_size)
    time_axis = np.arange(1, len(x) + 1)
    x = x.copy()
    x = x / np.sqrt(np.sum(x ** 2))

    spectrum = np.fft.fft(x, fft_size)
    weighted_spectrum = np.fft.fft(-x * 1j * time_axis, fft_size)
    centroid = -(weighted_spectrum.imag) * spectrum.real + spectrum.imag * weighted_spectrum.real
    return centroid


def d4c_dc_correction(signal, fs, fft_size, f0):
    fft_size = int(fft_size)
    frequency_axis = np.arange(fft_size) / fft_size * fs
    low_frequency_axis = frequency_axis[frequency_axis < f0 + fs / fft_size]
    low_frequency_replica = interp1d(f0 - low_frequency_axis,
            signal[frequency_axis < f0 + fs / fft_size],
            kind="linear",
            fill_value="extrapolate")(low_frequency_axis)
    idx = frequency_axis < f0
    signal[idx] = low_frequency_replica[idx[:len(low_frequency_replica)]] + signal[idx]
    signal[int(fft_size / 2.) + 1:] = signal[1 : int(fft_size / 2.)][::-1]
    return signal


def d4c_linear_smoothing(group_delay, fs, fft_size, width):
    double_frequency_axis = np.arange(2 * fft_size) / float(fft_size ) * fs - fs
    double_spectrum = np.concatenate([group_delay, group_delay])

    double_segment = np.cumsum(double_spectrum * (fs / float(fft_size)))
    center_frequency = np.arange(int(fft_size / 2) + 1) / float(fft_size ) * fs
    low_levels = cheaptrick_interp1h(double_frequency_axis + fs / float(fft_size) / 2.,
            double_segment, center_frequency - width / 2.)
    high_levels = cheaptrick_interp1h(double_frequency_axis + fs / float(fft_size) / 2.,
            double_segment, center_frequency + width / 2.)
    smoothed_spectrum = (high_levels - low_levels) / width
    return smoothed_spectrum


def d4c_get_smoothed_power_spectrum(waveform, fs, f0, fft_size):
    power_spectrum = np.abs(np.fft.fft(waveform, int(fft_size))) ** 2
    spectral_envelope = d4c_dc_correction(power_spectrum, fs, fft_size, f0)
    spectral_envelope = d4c_linear_smoothing(spectral_envelope, fs, fft_size, f0)
    spectral_envelope = np.concatenate([spectral_envelope,
        spectral_envelope[1:-1][::-1]])
    return spectral_envelope


def d4c_get_static_group_delay(static_centroid, smoothed_power_spectrum, fs, f0,
        fft_size):
    group_delay = static_centroid / smoothed_power_spectrum
    group_delay = d4c_linear_smoothing(group_delay, fs, fft_size, f0 / 2.)
    group_delay = np.concatenate([group_delay, group_delay[1:-1][::-1]])
    smoothed_group_delay = d4c_linear_smoothing(group_delay, fs, fft_size, f0)
    group_delay = group_delay[:int(fft_size / 2) + 1] - smoothed_group_delay
    group_delay = np.concatenate([group_delay, group_delay[1:-1][::-1]])
    return group_delay


def d4c_get_coarse_aperiodicity(group_delay, fs, fft_size,
        frequency_interval, number_of_aperiodicities, window1):
    boundary = np.round(fft_size / len(window1) * 8)
    half_window_length = np.floor(len(window1) / 2)
    coarse_aperiodicity = np.zeros((number_of_aperiodicities, 1))
    for i in range(1, number_of_aperiodicities + 1):
        center = np.floor(frequency_interval * i / (fs / float(fft_size)))
        segment = group_delay[int(center - half_window_length):int(center + half_window_length + 1)] * window1
        power_spectrum = np.abs(np.fft.fft(segment, int(fft_size))) ** 2
        cumulative_power_spectrum = np.cumsum(np.sort(power_spectrum[:int(fft_size / 2) + 1]))
        coarse_aperiodicity[i - 1] = -10 * np.log10(cumulative_power_spectrum[int(fft_size / 2 - boundary) - 1] / cumulative_power_spectrum[-1])
    return coarse_aperiodicity


def d4c_estimate_one_slice(x, fs, current_f0, frequency_interval,
        current_position, fft_size, number_of_aperiodicities, window1):
    if current_f0 == 0:
        coarse_aperiodicity = np.zeros((number_of_aperiodicities, 1))
        return coarse_aperiodicity

    static_centroid = d4c_get_static_centroid(x, fs, current_f0,
        current_position, fft_size)
    waveform = d4c_get_windowed_waveform(x, fs, current_f0, current_position,
            2, 1)
    smoothed_power_spectrum = d4c_get_smoothed_power_spectrum(waveform, fs,
            current_f0, fft_size)
    static_group_delay = d4c_get_static_group_delay(static_centroid,
            smoothed_power_spectrum, fs, current_f0, fft_size)
    coarse_aperiodicity = d4c_get_coarse_aperiodicity(static_group_delay,
            fs, fft_size, frequency_interval, number_of_aperiodicities, window1)
    return coarse_aperiodicity


def d4c(x, fs, temporal_positions_h, f0_h, vuv_h, threshold="default",
        fft_size="auto"):
    f0_low_limit = 47
    if fft_size == "auto":
        fft_size = 2 ** np.ceil(np.log2(4. * fs / f0_low_limit + 1.))
    else:
        raise ValueError("Only fft_size auto currently supported")
    f0_low_limit_for_spectrum = 71
    fft_size_for_spectrum = 2 ** np.ceil(np.log2(3 * fs / f0_low_limit_for_spectrum + 1.))
    threshold = 0.85
    upper_limit = 15000
    frequency_interval = 3000
    f0 = f0_h.copy()
    temporal_positions = temporal_positions_h.copy()
    f0[vuv_h == 0] = 0.

    number_of_aperiodicities = int(np.floor(np.min([upper_limit, fs / 2. - frequency_interval]) / float(frequency_interval)))
    window_length = np.floor(frequency_interval / (fs / float(fft_size))) * 2 + 1
    window1 =  harvest_nuttall(window_length)
    aperiodicity = np.zeros((int(fft_size_for_spectrum / 2) + 1, len(f0)))
    coarse_ap = np.zeros((1, len(f0)))

    frequency_axis = np.arange(int(fft_size_for_spectrum / 2) + 1) * float(fs) / fft_size_for_spectrum
    coarse_axis = np.arange(number_of_aperiodicities + 2) * frequency_interval
    coarse_axis[-1] = fs / 2.

    for i in range(len(f0)):
        r = d4c_love_train(x, fs, f0[i], temporal_positions_h[i], threshold)
        if r == 0:
            aperiodicity[:, i] = 1 - 0.000000000001
            continue
        current_f0 = max([f0_low_limit, f0[i]])
        coarse_aperiodicity = d4c_estimate_one_slice(x, fs, current_f0,
            frequency_interval, temporal_positions[i], fft_size,
            number_of_aperiodicities, window1)
        coarse_ap[0, i] = coarse_aperiodicity.ravel()[0]
        coarse_aperiodicity = np.maximum(0, coarse_aperiodicity - (current_f0 - 100) * 2. / 100.)
        piece = np.concatenate([[-60], -coarse_aperiodicity.ravel(), [-0.000000000001]])
        part = interp1d(coarse_axis, piece, kind="linear")(frequency_axis) / 20.
        aperiodicity[:, i] = 10 ** part
    return temporal_positions_h, f0_h, vuv_h, aperiodicity.T, coarse_ap.squeeze()


def world_synthesis_time_base_generation(temporal_positions, f0, fs, vuv,
        time_axis, default_f0):
    f0_interpolated_raw = interp1d(temporal_positions, f0, kind="linear",
            fill_value="extrapolate")(time_axis)
    vuv_interpolated = interp1d(temporal_positions, vuv, kind="linear",
            fill_value="extrapolate")(time_axis)
    vuv_interpolated = vuv_interpolated > 0.5
    f0_interpolated = f0_interpolated_raw * vuv_interpolated.astype("float32")
    f0_interpolated[f0_interpolated == 0] = f0_interpolated[f0_interpolated == 0] + default_f0
    total_phase = np.cumsum(2 * np.pi * f0_interpolated / float(fs))

    core = np.mod(total_phase, 2 * np.pi)
    core = np.abs(core[1:] - core[:-1])
    # account for diff, avoid deprecation warning with [:-1]
    pulse_locations = time_axis[:-1][core > (np.pi / 2.)]
    pulse_locations_index = np.round(pulse_locations * fs).astype("int32")
    return pulse_locations, pulse_locations_index, vuv_interpolated


def world_synthesis_get_spectral_parameters(temporal_positions,
        temporal_position_index, spectrogram, amplitude_periodic,
        amplitude_random, pulse_locations):
    floor_index = int(np.floor(temporal_position_index) - 1)
    assert floor_index >= 0
    ceil_index = int(np.ceil(temporal_position_index) - 1)
    t1 = temporal_positions[floor_index]
    t2 = temporal_positions[ceil_index]

    if t1 == t2:
        spectrum_slice = spectrogram[:, floor_index]
        periodic_slice = amplitude_periodic[:, floor_index]
        aperiodic_slice = amplitude_random[:, floor_index]
    else:
        cs = np.concatenate([spectrogram[:, floor_index][None],
            spectrogram[:, ceil_index][None]], axis=0)
        mmm = max([t1, min([t2, pulse_locations])])
        spectrum_slice = interp1d(np.array([t1, t2]), cs,
            kind="linear", axis=0)(mmm.copy())
        cp = np.concatenate([amplitude_periodic[:, floor_index][None],
            amplitude_periodic[:, ceil_index][None]], axis=0)
        periodic_slice = interp1d(np.array([t1, t2]), cp,
            kind="linear", axis=0)(mmm.copy())
        ca = np.concatenate([amplitude_random[:, floor_index][None],
            amplitude_random[:, ceil_index][None]], axis=0)
        aperiodic_slice = interp1d(np.array([t1, t2]), ca,
            kind="linear", axis=0)(mmm.copy())
    return spectrum_slice, periodic_slice, aperiodic_slice

"""
Filter data with an FIR filter using the overlap-add method.
from http://projects.scipy.org/scipy/attachment/ticket/837/fftfilt.py
"""
def nextpow2(x):
    """Return the first integer N such that 2**N >= abs(x)"""
    return np.ceil(np.log2(np.abs(x)))


def fftfilt(b, x, *n):
    """Filter the signal x with the FIR filter described by the
    coefficients in b using the overlap-add method. If the FFT
    length n is not specified, it and the overlap-add block length
    are selected so as to minimize the computational cost of
    the filtering operation."""

    N_x = len(x)
    N_b = len(b)

    # Determine the FFT length to use:
    if len(n):
        # Use the specified FFT length (rounded up to the nearest
        # power of 2), provided that it is no less than the filter
        # length:
        n = n[0]
        if n != int(n) or n <= 0:
            raise ValueError('n must be a nonnegative integer')
        if n < N_b:
            n = N_b
        N_fft = 2**nextpow2(n)
    else:
        if N_x > N_b:
            # When the filter length is smaller than the signal,
            # choose the FFT length and block size that minimize the
            # FLOPS cost. Since the cost for a length-N FFT is
            # (N/2)*log2(N) and the filtering operation of each block
            # involves 2 FFT operations and N multiplications, the
            # cost of the overlap-add method for 1 length-N block is
            # N*(1+log2(N)). For the sake of efficiency, only FFT
            # lengths that are powers of 2 are considered:
            N = 2**np.arange(np.ceil(np.log2(N_b)),
                             np.floor(np.log2(N_x)))
            cost = np.ceil(N_x/(N-N_b+1))*N*(np.log2(N)+1)
            N_fft = N[np.argmin(cost)]
        else:
            # When the filter length is at least as long as the signal,
            # filter the signal using a single block:
            N_fft = 2**nextpow2(N_b+N_x-1)

    N_fft = int(N_fft)

    # Compute the block length:
    L = int(N_fft - N_b + 1)

    # Compute the transform of the filter:
    H = np.fft.fft(b, N_fft)

    y = np.zeros(N_x, dtype=np.float32)
    i = 0
    while i <= N_x:
        il = min([i+L,N_x])
        k = min([i+N_fft,N_x])
        yt = np.fft.ifft(np.fft.fft(x[i:il],N_fft)*H,N_fft) # Overlap..
        y[i:k] = y[i:k] + yt[:k-i]            # and add
        i += L
    return y


def world_synthesis(f0_d4c, vuv_d4c, aperiodicity_d4c,
        spectrogram_ct, fs_ct, random_seed=1999):

    # swap 0 and 1 axis
    spectrogram_ct = spectrogram_ct.T
    fs = fs_ct
    # coarse -> fine aper
    if len(aperiodicity_d4c.shape) == 1 or aperiodicity_d4c.shape[1] == 1:
        print("Coarse aperiodicity detected - interpolating to full size")
        aper = np.zeros_like(spectrogram_ct)
        if len(aperiodicity_d4c.shape) == 1:
            aperiodicity_d4c = aperiodicity_d4c[None, :]
        else:
            aperiodicity_d4c = aperiodicity_d4c.T
        coarse_aper_d4c = aperiodicity_d4c
        frequency_interval = 3000
        upper_limit = 15000
        number_of_aperiodicities = int(np.floor(np.min([upper_limit, fs / 2. - frequency_interval]) / float(frequency_interval)))
        coarse_axis = np.arange(number_of_aperiodicities + 2) * frequency_interval
        coarse_axis[-1] = fs / 2.
        f0_low_limit_for_spectrum = 71
        fft_size_for_spectrum = 2 ** np.ceil(np.log2(3 * fs / f0_low_limit_for_spectrum + 1.))

        frequency_axis = np.arange(int(fft_size_for_spectrum / 2) + 1) * float(fs) / fft_size_for_spectrum

        for i in range(len(f0_d4c)):
            ca = coarse_aper_d4c[0, i]
            cf = f0_d4c[i]
            coarse_aperiodicity = np.maximum(0, ca - (cf - 100) * 2. / 100.)
            piece = np.concatenate([[-60], -ca.ravel(), [-0.000000000001]])
            part = interp1d(coarse_axis, piece, kind="linear")(frequency_axis) / 20.
            aper[:, i] = 10 ** part
        aperiodicity_d4c = aper
    else:
        aperiodicity_d4c = aperiodicity_d4c.T

    default_f0 = 500.
    random_state = np.random.RandomState(1999)
    spectrogram = spectrogram_ct
    aperiodicity = aperiodicity_d4c
    # max 30s, if greater than thrown an error
    max_len = 5000000
    _, temporal_positions = _world_get_temporal_positions(max_len, fs)
    temporal_positions = temporal_positions[:spectrogram.shape[1]]
    #temporal_positions = temporal_positions_d4c
    #from IPython import embed; embed()
    #raise ValueError()
    vuv = vuv_d4c
    f0 = f0_d4c

    time_axis = np.arange(temporal_positions[0], temporal_positions[-1],
            1. / fs)
    y = 0. * time_axis
    r = world_synthesis_time_base_generation(temporal_positions, f0, fs, vuv,
            time_axis, default_f0)
    pulse_locations, pulse_locations_index, interpolated_vuv = r
    fft_size = int((len(spectrogram) - 1) * 2)
    base_index = np.arange(-fft_size / 2, fft_size / 2) + 1
    y_length = len(y)
    tmp_complex_cepstrum = np.zeros((fft_size,), dtype=np.complex128)
    latter_index = np.arange(int(fft_size / 2) + 1, fft_size + 1) - 1

    temporal_position_index = interp1d(temporal_positions, np.arange(1, len(temporal_positions) + 1), kind="linear", fill_value="extrapolate")(pulse_locations)
    temporal_postion_index = np.maximum(1, np.minimum(len(temporal_positions),
        temporal_position_index)) - 1

    amplitude_aperiodic = aperiodicity ** 2
    amplitude_periodic = np.maximum(0.001, (1. - amplitude_aperiodic))

    for i in range(len(pulse_locations_index)):
        spectrum_slice, periodic_slice, aperiodic_slice = world_synthesis_get_spectral_parameters(
            temporal_positions, temporal_position_index[i], spectrogram,
            amplitude_periodic, amplitude_aperiodic, pulse_locations[i])
        idx = min(len(pulse_locations_index), i + 2) - 1
        noise_size = pulse_locations_index[idx] - pulse_locations_index[i]
        output_buffer_index = np.maximum(1, np.minimum(y_length, pulse_locations_index[i] + 1 + base_index)).astype("int32") - 1

        if interpolated_vuv[pulse_locations_index[i]] >= 0.5:
            tmp_periodic_spectrum = spectrum_slice * periodic_slice
            # eps in matlab/octave
            tmp_periodic_spectrum[tmp_periodic_spectrum == 0] = 2.2204E-16
            periodic_spectrum = np.concatenate([tmp_periodic_spectrum,
                tmp_periodic_spectrum[1:-1][::-1]])
            tmp_cepstrum = np.real(np.fft.fft(np.log(np.abs(periodic_spectrum)) / 2.))
            tmp_complex_cepstrum[latter_index] = tmp_cepstrum[latter_index] * 2
            tmp_complex_cepstrum[0] = tmp_cepstrum[0]

            response = np.fft.fftshift(np.real(np.fft.ifft(np.exp(np.fft.ifft(
                tmp_complex_cepstrum)))))
            y[output_buffer_index] += response * np.sqrt(
                   max([1, noise_size]))
            tmp_aperiodic_spectrum = spectrum_slice * aperiodic_slice
        else:
            tmp_aperiodic_spectrum = spectrum_slice

        tmp_aperiodic_spectrum[tmp_aperiodic_spectrum == 0] = 2.2204E-16
        aperiodic_spectrum = np.concatenate([tmp_aperiodic_spectrum,
            tmp_aperiodic_spectrum[1:-1][::-1]])
        tmp_cepstrum = np.real(np.fft.fft(np.log(np.abs(aperiodic_spectrum)) / 2.))
        tmp_complex_cepstrum[latter_index] = tmp_cepstrum[latter_index] * 2
        tmp_complex_cepstrum[0] = tmp_cepstrum[0]
        rc = np.fft.ifft(tmp_complex_cepstrum)
        erc = np.exp(rc)
        response = np.fft.fftshift(np.real(np.fft.ifft(erc)))
        noise_input = random_state.randn(max([3, noise_size]),)

        y[output_buffer_index] = y[output_buffer_index] + fftfilt(noise_input - np.mean(noise_input), response)
    return y


def _mgc_b2c(wc, c, alpha):
    wc_o = np.zeros_like(wc)
    desired_order = len(wc) - 1
    for i in range(0, len(c))[::-1]:
        prev = copy.copy(wc_o)
        wc_o[0] = c[i]
        if desired_order >= 1:
            wc_o[1] = (1. - alpha ** 2) * prev[0] + alpha * prev[1]
        for m in range(2, desired_order + 1):
            wc_o[m] = prev[m - 1] + alpha * (prev[m] - wc_o[m - 1])
    return wc_o


def _mgc_ptrans(p, m, alpha):
    d = 0.
    o = 0.

    d = p[m]
    for i in range(1, m)[::-1]:
        o = p[i] + alpha * d
        d = p[i]
        p[i] = o

    o = alpha * d
    p[0] = (1. - alpha ** 2) * p[0] + 2 * o


def _mgc_qtrans(q, m, alpha):
    d = q[1]
    for i in range(2, 2 * m + 1):
        o = q[i] + alpha * d
        d = q[i]
        q[i] = o


def _mgc_gain(er, c, m, g):
    t = 0.
    if g != 0:
        for i in range(1, m + 1):
            t += er[i] * c[i]
        return er[0] + g * t
    else:
        return er[0]


def _mgc_fill_toeplitz(A, t):
    n = len(t)
    for i in range(n):
        for j in range(n):
            A[i, j] = t[i - j] if i - j >= 0 else t[j - i]


def _mgc_fill_hankel(A, t):
    n = len(t) // 2 + 1
    for i in range(n):
        for j in range(n):
            A[i, j] = t[i + j]


def _mgc_ignorm(c, gamma):
    if gamma == 0.:
        c[0] = np.log(c[0])
        return c
    gain = c[0] ** gamma
    c[1:] *= gain
    c[0] = (gain - 1.) / gamma


def _mgc_gnorm(c, gamma):
    if gamma == 0.:
        c[0] = np.exp(c[0])
        return c
    gain = 1. + gamma * c[0]
    c[1:] /= gain
    c[0] = gain ** (1. / gamma)


def _mgc_b2mc(mc, alpha):
    m = len(mc)
    o = 0.
    d = mc[m - 1]
    for i in range(m - 1)[::-1]:
        o = mc[i] + alpha * d
        d = mc[i]
        mc[i] = o


def _mgc_mc2b(mc, alpha):
    itr = range(len(mc) - 1)[::-1]
    for i in itr:
        mc[i] = mc[i] - alpha * mc[i + 1]


def _mgc_gc2gc(src_ceps, src_gamma=0., dst_order=None, dst_gamma=0.):
    if dst_order == None:
        dst_order = len(src_ceps) - 1

    dst_ceps = np.zeros((dst_order + 1,), dtype=src_ceps.dtype)
    dst_order = len(dst_ceps) - 1
    m1 = len(src_ceps) - 1
    dst_ceps[0] = copy.deepcopy(src_ceps[0])

    for m in range(2, dst_order + 2):
        ss1 = 0.
        ss2 = 0.
        min_1 = m1 if (m1 < m - 1) else m - 2
        itr = range(2, min_1 + 2)
        if len(itr) < 1:
            if min_1 + 1 == 2:
                itr = [2]
            else:
                itr = []

        """
        # old slower version
        for k in itr:
            assert k >= 1
            assert (m - k) >= 0
            cc = src_ceps[k - 1] * dst_ceps[m - k]
            ss2 += (k - 1) * cc
            ss1 += (m - k) * cc
        """

        if len(itr) > 0:
            itr = np.array(itr)
            cc_a = src_ceps[itr - 1] * dst_ceps[m - itr]
            ss2 += ((itr - 1) * cc_a).sum()
            ss1 += ((m - itr) * cc_a).sum()

        if m <= m1 + 1:
            dst_ceps[m - 1] = src_ceps[m - 1] + (dst_gamma * ss2 - src_gamma * ss1)/(m - 1.)
        else:
            dst_ceps[m - 1] = (dst_gamma * ss2 - src_gamma * ss1) / (m - 1.)
    return dst_ceps


def _mgc_newton(mgc_stored, periodogram, order, alpha, gamma,
        recursion_order, iter_number, y_fft, z_fft, cr, pr, rr, ri,
        qr, qi, Tm, Hm, Tm_plus_Hm, b):
    # a lot of inplace operations to match the Julia code
    cr[1:order + 1] = mgc_stored[1:order + 1]

    if alpha != 0:
        cr_res = _mgc_b2c(cr[:recursion_order + 1], cr[:order + 1], -alpha)
        cr[:recursion_order + 1] = cr_res[:]

    y = sp.fftpack.fft(np.cast["float64"](cr))
    c = mgc_stored
    x = periodogram
    if gamma != 0.:
        gamma_inv = 1. / gamma
    else:
        gamma_inv = np.inf

    if gamma == -1.:
        pr[:] = copy.deepcopy(x)
        new_pr = copy.deepcopy(pr)
    elif gamma == 0.:
        pr[:] = copy.deepcopy(x) / np.exp(2 * np.real(y))
        new_pr = copy.deepcopy(pr)
    else:
        tr = 1. + gamma * np.real(y)
        ti = -gamma * np.imag(y)
        trr = tr * tr
        tii = ti * ti
        s = trr + tii
        t = x * np.power(s, (-gamma_inv))
        t /= s
        pr[:] = t
        rr[:] = tr * t
        ri[:] = ti * t
        t /= s
        qr[:] = (trr - tii) * t
        s = tr * ti * t
        qi[:] = (s + s)
        new_pr = copy.deepcopy(pr)

    if gamma != -1.:
        """
        print()
        print(pr.sum())
        print(rr.sum())
        print(ri.sum())
        print(qr.sum())
        print(qi.sum())
        print()
        """
        pass

    y_fft[:] = copy.deepcopy(pr) + 0.j
    z_fft[:] = np.fft.fft(y_fft) / len(y_fft)
    pr[:] = copy.deepcopy(np.real(z_fft))
    if alpha != 0.:
        idx_1 = pr[:2 * order + 1]
        idx_2 = pr[:recursion_order + 1]
        idx_3 = _mgc_b2c(idx_1, idx_2, alpha)
        pr[:2 * order + 1] = idx_3[:]

    if gamma == 0. or gamma == -1.:
        qr[:2 * order + 1] = pr[:2 * order + 1]
        rr[:order + 1] = copy.deepcopy(pr[:order + 1])
    else:
        for i in range(len(qr)):
            y_fft[i] = qr[i] + 1j * qi[i]
        z_fft[:] = np.fft.fft(y_fft) / len(y_fft)
        qr[:] = np.real(z_fft)

        for i in range(len(rr)):
            y_fft[i] = rr[i] + 1j * ri[i]
        z_fft[:] = np.fft.fft(y_fft) / len(y_fft)
        rr[:] = np.real(z_fft)

        if alpha != 0.:
            qr_new = _mgc_b2c(qr[:recursion_order + 1], qr[:recursion_order + 1], alpha)
            qr[:recursion_order + 1] = qr_new[:]
            rr_new = _mgc_b2c(rr[:order + 1], rr[:recursion_order + 1], alpha)
            rr[:order + 1] = rr_new[:]

    if alpha != 0:
        _mgc_ptrans(pr, order, alpha)
        _mgc_qtrans(qr, order, alpha)

    eta = 0.
    if gamma != -1.:
        eta = _mgc_gain(rr, c, order, gamma)
        c[0] = np.sqrt(eta)

    if gamma == -1.:
        qr[:] = 0.
    elif gamma != 0.:
        for i in range(2, 2 * order + 1):
            qr[i] *= 1. + gamma

    te = pr[:order]
    _mgc_fill_toeplitz(Tm, te)
    he = qr[2: 2 * order + 1]
    _mgc_fill_hankel(Hm, he)

    Tm_plus_Hm[:] = Hm[:] + Tm[:]
    b[:order] = rr[1:order + 1]
    res = np.linalg.solve(Tm_plus_Hm, b)
    b[:] = res[:]

    c[1:order + 1] += res[:order]

    if gamma == -1.:
        eta = _mgc_gain(rr, c, order, gamma)
        c[0] = np.sqrt(eta)
    return np.log(eta), new_pr


def _mgc_mgcepnorm(b_gamma, alpha, gamma, otype):
    if otype != 0:
        raise ValueError("Not yet implemented for otype != 0")

    mgc = copy.deepcopy(b_gamma)
    _mgc_ignorm(mgc, gamma)
    _mgc_b2mc(mgc, alpha)
    return mgc


def _sp2mgc(sp, order=20, alpha=0.35, gamma=-0.41, miniter=2, maxiter=30, criteria=0.001, otype=0, verbose=False):
    # Based on r9y9 Julia code
    # https://github.com/r9y9/MelGeneralizedCepstrums.jl
    periodogram = np.abs(sp) ** 2
    recursion_order = len(periodogram) - 1
    slen = len(periodogram)
    iter_number = 1

    def _z():
        return np.zeros((slen,), dtype="float64")

    def _o():
        return np.zeros((order,), dtype="float64")

    def _o2():
        return np.zeros((order, order), dtype="float64")

    cr = _z()
    pr = _z()
    rr = _z()
    ri = _z().astype("float128")
    qr = _z()
    qi = _z().astype("float128")
    Tm = _o2()
    Hm = _o2()
    Tm_plus_Hm = _o2()
    b = _o()
    y = _z() + 0j
    z = _z() + 0j
    b_gamma = np.zeros((order + 1,), dtype="float64")
    # return pr_new due to oddness with Julia having different numbers
    # in pr at end of function vs back in this scope
    eta0, pr_new = _mgc_newton(b_gamma, periodogram, order, alpha, -1.,
                               recursion_order, iter_number, y, z, cr, pr, rr,
                               ri, qr, qi, Tm, Hm, Tm_plus_Hm, b)
    pr[:] = pr_new
    """
    print(eta0)
    print(sum(b_gamma))
    print(sum(periodogram))
    print(order)
    print(alpha)
    print(recursion_order)
    print(sum(y))
    print(sum(cr))
    print(sum(z))
    print(sum(pr))
    print(sum(rr))
    print(sum(qi))
    print(Tm.sum())
    print(Hm.sum())
    print(sum(b))
    raise ValueError()
    """
    if gamma != -1.:
        d = np.zeros((order + 1,), dtype="float64")
        if alpha != 0.:
            _mgc_ignorm(b_gamma, -1.)
            _mgc_b2mc(b_gamma, alpha)
            d = copy.deepcopy(b_gamma)
            _mgc_gnorm(d, -1.)
            # numbers are slightly different here - numerical diffs?
        else:
            d = copy.deepcopy(b_gamma)
        b_gamma = _mgc_gc2gc(d, -1., order, gamma)

        if alpha != 0.:
            _mgc_ignorm(b_gamma, gamma)
            _mgc_mc2b(b_gamma, alpha)
            _mgc_gnorm(b_gamma, gamma)

    if gamma != -1.:
        eta_t = eta0
        for i in range(1, maxiter + 1):
            eta, pr_new = _mgc_newton(b_gamma, periodogram, order, alpha,
                    gamma, recursion_order, i, y, z, cr, pr, rr,
                    ri, qr, qi, Tm, Hm, Tm_plus_Hm, b)
            pr[:] = pr_new
            """
            print(eta0)
            print(sum(b_gamma))
            print(sum(periodogram))
            print(order)
            print(alpha)
            print(recursion_order)
            print(sum(y))
            print(sum(cr))
            print(sum(z))
            print(sum(pr))
            print(sum(rr))
            print(sum(qi))
            print(Tm.sum())
            print(Hm.sum())
            print(sum(b))
            raise ValueError()
            """
            err = np.abs((eta_t - eta) / eta)
            if verbose:
                print("iter %i, criterion: %f" % (i, err))
            if i >= miniter:
                if err < criteria:
                    if verbose:
                        print("optimization complete at iter %i" % i)
                    break
            eta_t = eta
    mgc_arr = _mgc_mgcepnorm(b_gamma, alpha, gamma, otype)
    return mgc_arr


_sp_convert_results = []

def _sp_collect_result(result):
    _sp_convert_results.append(result)


def _sp_convert(c_i, order, alpha, gamma, miniter, maxiter, criteria,
        otype, verbose):
    i = c_i[0]
    tot_i = c_i[1]
    sp_i = c_i[2]
    r_i = (i, _sp2mgc(sp_i, order=order, alpha=alpha, gamma=gamma,
                miniter=miniter, maxiter=maxiter, criteria=criteria,
                otype=otype, verbose=verbose))
    return r_i


def sp2mgc(sp, order=20, alpha=0.35, gamma=-0.41, miniter=2,
        maxiter=30, criteria=0.001, otype=0, verbose=False):
    """
    Accepts 1D or 2D one-sided spectrum (complex or real valued).

    If 2D, assumes time is axis 0.

    Returns mel generalized cepstral coefficients.

    Based on r9y9 Julia code
    https://github.com/r9y9/MelGeneralizedCepstrums.jl
    """

    if len(sp.shape) == 1:
        sp = np.concatenate((sp, sp[:, 1:][:, ::-1]), axis=0)
        return _sp2mgc(sp, order=order, alpha=alpha, gamma=gamma,
                miniter=miniter, maxiter=maxiter, criteria=criteria,
                otype=otype, verbose=verbose)
    else:
        sp = np.concatenate((sp, sp[:, 1:][:, ::-1]), axis=1)
        # Slooow, use multiprocessing to speed up a bit
        # http://blog.shenwei.me/python-multiprocessing-pool-difference-between-map-apply-map_async-apply_async/
        # http://stackoverflow.com/questions/5666576/show-the-progress-of-a-python-multiprocessing-pool-map-call
        c = [(i + 1, sp.shape[0], sp[i]) for i in range(sp.shape[0])]
        p = Pool()
        start = time.time()
        if verbose:
            print("Starting conversion of %i frames" % sp.shape[0])
            print("This may take some time...")

        # takes ~360s for 630 frames, 1 process
        itr = p.map_async(functools.partial(_sp_convert, order=order, alpha=alpha, gamma=gamma, miniter=miniter, maxiter=maxiter, criteria=criteria, otype=otype, verbose=False), c, callback=_sp_collect_result)

        sz = len(c) // itr._chunksize
        if (sz * itr._chunksize) != len(c):
            sz += 1

        last_remaining = None
        while True:
            remaining = itr._number_left
            if verbose:
                if remaining != last_remaining:
                    last_remaining = remaining
                    print("%i chunks of %i complete" % (sz - remaining, sz))
            if itr.ready():
                break
            time.sleep(.5)

        """
        # takes ~455s for 630 frames
        itr = p.imap_unordered(functools.partial(_sp_convert, order=order, alpha=alpha, gamma=gamma, miniter=miniter, maxiter=maxiter, criteria=criteria, otype=otype, verbose=False), c)
        res = []
        # print ~every 5%
        mod = int(len(c)) // 20
        if mod < 1:
            mod = 1
        for i, res_i in enumerate(itr, 1):
            res.append(res_i)
            if i % mod == 0 or i == 1:
                print("%i of %i complete" % (i, len(c)))
        """
        p.close()
        p.join()
        stop = time.time()
        if verbose:
            print("Processed %i frames in %s seconds" % (sp.shape[0], stop - start))
        # map_async result comes in chunks
        flat = [a_i for a in _sp_convert_results for a_i in a]
        final = [o[1] for o in sorted(flat, key=lambda x: x[0])]
        for i in range(len(_sp_convert_results)):
            _sp_convert_results.pop()
        return np.array(final)


def win2mgc(windowed_signal, order=20, alpha=0.35, gamma=-0.41, miniter=2,
        maxiter=30, criteria=0.001, otype=0, verbose=False):
    """
    Accepts 1D or 2D array of windowed signal frames.

    If 2D, assumes time is axis 0.

    Returns mel generalized cepstral coefficients.

    Based on r9y9 Julia code
    https://github.com/r9y9/MelGeneralizedCepstrums.jl
    """
    if len(windowed_signal.shape) == 1:
        sp = np.fft.fft(windowed_signal)
        return _sp2mgc(sp, order=order, alpha=alpha, gamma=gamma,
                miniter=miniter, maxiter=maxiter, criteria=criteria,
                otype=otype, verbose=verbose)
    else:
        raise ValueError("2D input not yet complete for win2mgc")


def _mgc_freqt(wc, c, alpha):
    prev = np.zeros_like(wc)
    dst_order = len(wc) - 1
    wc *= 0
    m1 = len(c) - 1
    for i in range(-m1, 1, 1):
        prev[:] = wc
        if dst_order >= 0:
            wc[0] = c[-i] + alpha * prev[0]
        if dst_order >= 1:
            wc[1] = (1. - alpha * alpha) * prev[0] + alpha * prev[1]
        for m in range(2, dst_order + 1):
            wc[m] = prev[m - 1] + alpha * (prev[m] - wc[m - 1])


def _mgc_mgc2mgc(src_ceps, src_alpha, src_gamma, dst_order, dst_alpha, dst_gamma):
    dst_ceps = np.zeros((dst_order + 1,))
    alpha = (dst_alpha - src_alpha) / (1. - dst_alpha * src_alpha)
    if alpha == 0.:
        new_dst_ceps = copy.deepcopy(src_ceps)
        _mgc_gnorm(new_dst_ceps, src_gamma)
        dst_ceps = _mgc_gc2gc(new_dst_ceps, src_gamma, dst_order, dst_gamma)
        _mgc_ignorm(dst_ceps, dst_gamma)
    else:
        _mgc_freqt(dst_ceps, src_ceps, alpha)
        _mgc_gnorm(dst_ceps, src_gamma)
        new_dst_ceps = copy.deepcopy(dst_ceps)
        dst_ceps = _mgc_gc2gc(new_dst_ceps, src_gamma, dst_order, dst_gamma)
        _mgc_ignorm(dst_ceps, dst_gamma)
    return dst_ceps


_mgc_convert_results = []

def _mgc_collect_result(result):
    _mgc_convert_results.append(result)


def _mgc_convert(c_i, alpha, gamma, fftlen):
    i = c_i[0]
    tot_i = c_i[1]
    mgc_i = c_i[2]
    r_i = (i, _mgc_mgc2mgc(mgc_i, src_alpha=alpha, src_gamma=gamma,
                dst_order=fftlen // 2, dst_alpha=0., dst_gamma=0.))
    return r_i


def mgc2sp(mgc_arr, alpha=0.35, gamma=-0.41, fftlen="auto", fs=None,
        mode="world_pad", verbose=False):
    """
    Accepts 1D or 2D array of mgc

    If 2D, assume time is on axis 0

    Returns reconstructed smooth spectrum

    Based on r9y9 Julia code
    https://github.com/r9y9/MelGeneralizedCepstrums.jl
    """
    if mode != "world_pad":
        raise ValueError("Only currently supported mode is world_pad")

    if fftlen == "auto":
        if fs == None:
            raise ValueError("fs must be provided for fftlen 'auto'")
        f0_low_limit = 71
        fftlen = int(2 ** np.ceil(np.log2(3. * float(fs) / f0_low_limit + 1)))
        if verbose:
            print("setting fftlen to %i" % fftlen)

    if len(mgc_arr.shape) == 1:
        c = _mgc_mgc2mgc(mgc_arr, alpha, gamma, fftlen // 2, 0., 0.)
        buf = np.zeros((fftlen,), dtype=c.dtype)
        buf[:len(c)] = c[:]
        return np.fft.rfft(buf)
    else:
        # Slooow, use multiprocessing to speed up a bit
        # http://blog.shenwei.me/python-multiprocessing-pool-difference-between-map-apply-map_async-apply_async/
        # http://stackoverflow.com/questions/5666576/show-the-progress-of-a-python-multiprocessing-pool-map-call
        c = [(i + 1, mgc_arr.shape[0], mgc_arr[i]) for i in range(mgc_arr.shape[0])]
        p = Pool()
        start = time.time()
        if verbose:
            print("Starting conversion of %i frames" % mgc_arr.shape[0])
            print("This may take some time...")
        #itr = p.map(functools.partial(_mgc_convert, alpha=alpha, gamma=gamma, fftlen=fftlen), c)
        #raise ValueError()

        # 500.1 s for 630 frames process
        itr = p.map_async(functools.partial(_mgc_convert, alpha=alpha, gamma=gamma, fftlen=fftlen), c, callback=_mgc_collect_result)

        sz = len(c) // itr._chunksize
        if (sz * itr._chunksize) != len(c):
            sz += 1

        last_remaining = None
        while True:
            remaining = itr._number_left
            if verbose:
                if last_remaining != remaining:
                    last_remaining = remaining
                    print("%i chunks of %i complete" % (sz - remaining, sz))
            if itr.ready():
                break
            time.sleep(.5)
        p.close()
        p.join()
        stop = time.time()
        if verbose:
            print("Processed %i frames in %s seconds" % (mgc_arr.shape[0], stop - start))
        # map_async result comes in chunks
        flat = [a_i for a in _mgc_convert_results for a_i in a]
        final = [o[1] for o in sorted(flat, key=lambda x: x[0])]
        for i in range(len(_mgc_convert_results)):
            _mgc_convert_results.pop()
        c = np.array(final)
        buf = np.zeros((len(c), fftlen), dtype=c.dtype)
        buf[:, :c.shape[1]] = c[:]
        return np.exp(np.fft.rfft(buf, axis=-1).real)


def implot(arr, scale=None, title="", cmap="gray"):
    import matplotlib.pyplot as plt
    if scale is "specgram":
        # plotting part
        mag = 20. * np.log10(np.abs(arr))
        # Transpose so time is X axis, and invert y axis so
        # frequency is low at bottom
        mag = mag.T[::-1, :]
    else:
        mag = arr
    f, ax = plt.subplots()
    ax.matshow(mag, cmap=cmap)
    plt.axis("off")
    x1 = mag.shape[0]
    y1 = mag.shape[1]

    def autoaspect(x_range, y_range):
        """
        The aspect to make a plot square with ax.set_aspect in Matplotlib
        """
        mx = max(x_range, y_range)
        mn = min(x_range, y_range)
        if x_range <= y_range:
            return mx / float(mn)
        else:
            return mn / float(mx)
    asp = autoaspect(x1, y1)
    ax.set_aspect(asp)
    plt.title(title)


def test_lpc_to_lsf():
    # Matlab style vectors for testing
    # lsf = [0.7842 1.5605 1.8776 1.8984 2.3593]
    # a = [1.0000  0.6149  0.9899  0.0000  0.0031 -0.0082];
    lsf = [[0.7842, 1.5605, 1.8776, 1.8984, 2.3593],
           [0.7842, 1.5605, 1.8776, 1.8984, 2.3593]]
    a = [[1.0000, 0.6149, 0.9899, 0.0000, 0.0031, -0.0082],
         [1.0000, 0.6149, 0.9899, 0.0000, 0.0031, -0.0082]]
    a = np.array(a)
    lsf = np.array(lsf)
    lsf_r = lpc_to_lsf(a)
    assert_almost_equal(lsf, lsf_r, decimal=4)
    a_r = lsf_to_lpc(lsf)
    assert_almost_equal(a, a_r, decimal=4)
    lsf_r = lpc_to_lsf(a[0])
    assert_almost_equal(lsf[0], lsf_r, decimal=4)
    a_r = lsf_to_lpc(lsf[0])
    assert_almost_equal(a[0], a_r, decimal=4)


def test_lpc_analysis_truncate():
    # Test that truncate doesn't crash and actually truncates
    [a, g, e] = lpc_analysis(np.random.randn(85), order=8, window_step=80,
                             window_size=80, emphasis=0.9, truncate=True)
    assert(a.shape[0] == 1)


def test_feature_build():
    samplerate, X = fetch_sample_music()
    # MATLAB wavread does normalization
    X = X.astype('float32') / (2 ** 15)
    wsz = 256
    wst = 128
    a, g, e = lpc_analysis(X, order=8, window_step=wst,
                           window_size=wsz, emphasis=0.9,
                           copy=True)
    v, p = voiced_unvoiced(X, window_size=wsz,
                           window_step=wst)
    c = compress(e, n_components=64)
    # First component of a is always 1
    combined = np.hstack((a[:, 1:], g, c[:a.shape[0]]))
    features = np.zeros((a.shape[0], 2 * combined.shape[1]))
    start_indices = v * combined.shape[1]
    start_indices = start_indices.astype('int32')
    end_indices = (v + 1) * combined.shape[1]
    end_indices = end_indices.astype('int32')
    for i in range(features.shape[0]):
        features[i, start_indices[i]:end_indices[i]] = combined[i]


def test_mdct_and_inverse():
    fs, X = fetch_sample_music()
    X_dct = mdct_slow(X)
    X_r = imdct_slow(X_dct)
    assert np.all(np.abs(X_r[:len(X)] - X) < 1E-3)
    assert np.abs(X_r[:len(X)] - X).mean() < 1E-6


def test_all():
    test_lpc_analysis_truncate()
    test_feature_build()
    test_lpc_to_lsf()
    test_mdct_and_inverse()


def run_lpc_example():
    # ae.wav is from
    # http://www.linguistics.ucla.edu/people/hayes/103/Charts/VChart/ae.wav
    # Partially following the formant tutorial here
    # http://www.mathworks.com/help/signal/ug/formant-estimation-with-lpc-coefficients.html

    samplerate, X = fetch_sample_music()

    c = overlap_compress(X, 200, 400)
    X_r = overlap_uncompress(c, 400)
    wavfile.write('lpc_uncompress.wav', samplerate, soundsc(X_r))

    print("Calculating sinusoids")
    f_hz, m = sinusoid_analysis(X, input_sample_rate=16000)
    Xs_sine = sinusoid_synthesis(f_hz, m)
    orig_fname = 'lpc_orig.wav'
    sine_fname = 'lpc_sine_synth.wav'
    wavfile.write(orig_fname, samplerate, soundsc(X))
    wavfile.write(sine_fname, samplerate, soundsc(Xs_sine))

    lpc_order_list = [8, ]
    dct_components_list = [200, ]
    window_size_list = [400, ]
    # Seems like a dct component size of ~2/3rds the step
    # (1/3rd the window for 50% overlap) works well.
    for lpc_order in lpc_order_list:
        for dct_components in dct_components_list:
            for window_size in window_size_list:
                # 50% overlap
                window_step = window_size // 2
                a, g, e = lpc_analysis(X, order=lpc_order,
                                       window_step=window_step,
                                       window_size=window_size, emphasis=0.9,
                                       copy=True)
                print("Calculating LSF")
                lsf = lpc_to_lsf(a)
                # Not window_size - window_step! Need to implement overlap
                print("Calculating compression")
                c = compress(e, n_components=dct_components,
                             window_size=window_step)
                co = overlap_compress(e, n_components=dct_components,
                                      window_size=window_step)
                block_excitation = uncompress(c, window_size=window_step)
                overlap_excitation = overlap_uncompress(co,
                                                        window_size=window_step)
                a_r = lsf_to_lpc(lsf)
                f, m = lpc_to_frequency(a_r, g)
                block_lpc = lpc_synthesis(a_r, g, block_excitation,
                                          emphasis=0.9,
                                          window_step=window_step)
                overlap_lpc = lpc_synthesis(a_r, g, overlap_excitation,
                                            emphasis=0.9,
                                            window_step=window_step)
                v, p = voiced_unvoiced(X, window_size=window_size,
                                       window_step=window_step)
                noisy_lpc = lpc_synthesis(a_r, g, voiced_frames=v,
                                          emphasis=0.9,
                                          window_step=window_step)
                if dct_components is None:
                    dct_components = window_size
                noisy_fname = 'lpc_noisy_synth_%iwin_%ilpc_%idct.wav' % (
                    window_size, lpc_order, dct_components)
                block_fname = 'lpc_block_synth_%iwin_%ilpc_%idct.wav' % (
                    window_size, lpc_order, dct_components)
                overlap_fname = 'lpc_overlap_synth_%iwin_%ilpc_%idct.wav' % (
                    window_size, lpc_order, dct_components)
                wavfile.write(noisy_fname, samplerate, soundsc(noisy_lpc))
                wavfile.write(block_fname, samplerate,
                              soundsc(block_lpc))
                wavfile.write(overlap_fname, samplerate,
                              soundsc(overlap_lpc))


def run_fft_vq_example():
    n_fft = 512
    time_smoothing = 4
    def _pre(list_of_data):
        f_c = np.vstack([stft(dd, n_fft) for dd in list_of_data])
        if len(f_c) % time_smoothing != 0:
            newlen = len(f_c) - len(f_c) % time_smoothing
            f_c = f_c[:newlen]
        f_mag = complex_to_abs(f_c)
        f_phs = complex_to_angle(f_c)
        f_sincos = angle_to_sin_cos(f_phs)
        f_r = np.hstack((f_mag, f_sincos))
        f_r = f_r.reshape((len(f_r) // time_smoothing,
                           time_smoothing * f_r.shape[1]))
        return f_r, n_fft

    def preprocess_train(list_of_data, random_state):
        f_r, n_fft = _pre(list_of_data)
        clusters = f_r
        return clusters

    def apply_preprocess(list_of_data, clusters):
        f_r, n_fft = _pre(list_of_data)
        memberships, distances = vq(f_r, clusters)
        vq_r = clusters[memberships]
        vq_r = vq_r.reshape((time_smoothing * len(vq_r),
                             vq_r.shape[1] // time_smoothing))
        f_mag = vq_r[:, :n_fft // 2 + 1]
        f_sincos = vq_r[:, n_fft // 2 + 1:]
        extent = f_sincos.shape[1] // 2
        f_phs = sin_cos_to_angle(f_sincos[:, :extent], f_sincos[:, extent:])
        vq_c = abs_and_angle_to_complex(f_mag, f_phs)
        d_k = istft(vq_c, fftsize=n_fft)
        return d_k

    random_state = np.random.RandomState(1999)

    """
    fs, d = fetch_sample_music()
    sub = int(.8 * d.shape[0])
    d1 = [d[:sub]]
    d2 = [d[sub:]]
    """

    fs, d = fetch_sample_speech_fruit()
    d1 = d[::8] + d[1::8] + d[2::8] + d[3::8] + d[4::8] + d[5::8] + d[6::8]
    d2 = d[7::8]
    # make sure d1 and d2 aren't the same!
    assert [len(di) for di in d1] != [len(di) for di in d2]

    clusters = preprocess_train(d1, random_state)
    # Training data
    vq_d1 = apply_preprocess(d1, clusters)
    vq_d2 = apply_preprocess(d2, clusters)
    assert [i != j for i, j in zip(vq_d1.ravel(), vq_d2.ravel())]

    fix_d1 = np.concatenate(d1)
    fix_d2 = np.concatenate(d2)

    wavfile.write("fft_train_no_agc.wav", fs, soundsc(fix_d1))
    wavfile.write("fft_test_no_agc.wav", fs, soundsc(fix_d2))
    wavfile.write("fft_vq_train_no_agc.wav", fs, soundsc(vq_d1, fs))
    wavfile.write("fft_vq_test_no_agc.wav", fs, soundsc(vq_d2, fs))

    agc_d1, freq_d1, energy_d1 = time_attack_agc(fix_d1, fs, .5, 5)
    agc_d2, freq_d2, energy_d2 = time_attack_agc(fix_d2, fs, .5, 5)
    agc_vq_d1, freq_vq_d1, energy_vq_d1 = time_attack_agc(vq_d1, fs, .5, 5)
    agc_vq_d2, freq_vq_d2, energy_vq_d2 = time_attack_agc(vq_d2, fs, .5, 5)

    wavfile.write("fft_train_agc.wav", fs, soundsc(agc_d1))
    wavfile.write("fft_test_agc.wav", fs, soundsc(agc_d2))
    wavfile.write("fft_vq_train_agc.wav", fs, soundsc(agc_vq_d1, fs))
    wavfile.write("fft_vq_test_agc.wav", fs, soundsc(agc_vq_d2))


def run_dct_vq_example():
    def _pre(list_of_data):
        # Temporal window setting is crucial! - 512 seems OK for music, 256
        # fruit perhaps due to samplerates
        n_dct = 512
        f_r = np.vstack([mdct_slow(dd, n_dct) for dd in list_of_data])
        return f_r, n_dct

    def preprocess_train(list_of_data, random_state):
        f_r, n_dct = _pre(list_of_data)
        clusters = f_r
        return clusters

    def apply_preprocess(list_of_data, clusters):
        f_r, n_dct = _pre(list_of_data)
        f_clust = f_r
        memberships, distances = vq(f_clust, clusters)
        vq_r = clusters[memberships]
        d_k = imdct_slow(vq_r, n_dct)
        return d_k

    random_state = np.random.RandomState(1999)

    # This doesn't work very well due to only taking a sample from the end as
    # test
    fs, d = fetch_sample_music()
    sub = int(.8 * d.shape[0])
    d1 = [d[:sub]]
    d2 = [d[sub:]]

    """
    fs, d = fetch_sample_speech_fruit()
    d1 = d[::8] + d[1::8] + d[2::8] + d[3::8] + d[4::8] + d[5::8] + d[6::8]
    d2 = d[7::8]
    # make sure d1 and d2 aren't the same!
    assert [len(di) for di in d1] != [len(di) for di in d2]
    """

    clusters = preprocess_train(d1, random_state)
    # Training data
    vq_d1 = apply_preprocess(d1, clusters)
    vq_d2 = apply_preprocess(d2, clusters)
    assert [i != j for i, j in zip(vq_d2.ravel(), vq_d2.ravel())]

    fix_d1 = np.concatenate(d1)
    fix_d2 = np.concatenate(d2)

    wavfile.write("dct_train_no_agc.wav", fs, soundsc(fix_d1))
    wavfile.write("dct_test_no_agc.wav", fs, soundsc(fix_d2))
    wavfile.write("dct_vq_train_no_agc.wav", fs, soundsc(vq_d1))
    wavfile.write("dct_vq_test_no_agc.wav", fs, soundsc(vq_d2))

    """
    import matplotlib.pyplot as plt
    plt.specgram(vq_d2, cmap="gray")
    plt.figure()
    plt.specgram(fix_d2, cmap="gray")
    plt.show()
    """

    agc_d1, freq_d1, energy_d1 = time_attack_agc(fix_d1, fs, .5, 5)
    agc_d2, freq_d2, energy_d2 = time_attack_agc(fix_d2, fs, .5, 5)
    agc_vq_d1, freq_vq_d1, energy_vq_d1 = time_attack_agc(vq_d1, fs, .5, 5)
    agc_vq_d2, freq_vq_d2, energy_vq_d2 = time_attack_agc(vq_d2, fs, .5, 5)

    wavfile.write("dct_train_agc.wav", fs, soundsc(agc_d1))
    wavfile.write("dct_test_agc.wav", fs, soundsc(agc_d2))
    wavfile.write("dct_vq_train_agc.wav", fs, soundsc(agc_vq_d1))
    wavfile.write("dct_vq_test_agc.wav", fs, soundsc(agc_vq_d2))


def run_phase_reconstruction_example():
    fs, d = fetch_sample_speech_tapestry()
    # actually gives however many components you say! So double what .m file
    # says
    fftsize = 512
    step = 64
    X_s = np.abs(stft(d, fftsize=fftsize, step=step, real=False,
                      compute_onesided=False))
    X_t = iterate_invert_spectrogram(X_s, fftsize, step, verbose=True)

    """
    import matplotlib.pyplot as plt
    plt.specgram(d, cmap="gray")
    plt.savefig("1.png")
    plt.close()
    plt.imshow(X_s, cmap="gray")
    plt.savefig("2.png")
    plt.close()
    """

    wavfile.write("phase_original.wav", fs, soundsc(d))
    wavfile.write("phase_reconstruction.wav", fs, soundsc(X_t))


def run_phase_vq_example():
    def _pre(list_of_data):
        # Temporal window setting is crucial! - 512 seems OK for music, 256
        # fruit perhaps due to samplerates
        n_fft = 256
        step = 32
        f_r = np.vstack([np.abs(stft(dd, n_fft, step=step, real=False,
                                compute_onesided=False))
                         for dd in list_of_data])
        return f_r, n_fft, step

    def preprocess_train(list_of_data, random_state):
        f_r, n_fft, step = _pre(list_of_data)
        clusters = copy.deepcopy(f_r)
        return clusters

    def apply_preprocess(list_of_data, clusters):
        f_r, n_fft, step = _pre(list_of_data)
        f_clust = f_r
        # Nondeterministic ?
        memberships, distances = vq(f_clust, clusters)
        vq_r = clusters[memberships]
        d_k = iterate_invert_spectrogram(vq_r, n_fft, step, verbose=True)
        return d_k

    random_state = np.random.RandomState(1999)

    fs, d = fetch_sample_speech_fruit()
    d1 = d[::9]
    d2 = d[7::8][:5]
    # make sure d1 and d2 aren't the same!
    assert [len(di) for di in d1] != [len(di) for di in d2]

    clusters = preprocess_train(d1, random_state)
    fix_d1 = np.concatenate(d1)
    fix_d2 = np.concatenate(d2)
    vq_d2 = apply_preprocess(d2, clusters)

    wavfile.write("phase_train_no_agc.wav", fs, soundsc(fix_d1))
    wavfile.write("phase_vq_test_no_agc.wav", fs, soundsc(vq_d2))

    agc_d1, freq_d1, energy_d1 = time_attack_agc(fix_d1, fs, .5, 5)
    agc_d2, freq_d2, energy_d2 = time_attack_agc(fix_d2, fs, .5, 5)
    agc_vq_d2, freq_vq_d2, energy_vq_d2 = time_attack_agc(vq_d2, fs, .5, 5)

    """
    import matplotlib.pyplot as plt
    plt.specgram(agc_vq_d2, cmap="gray")
    #plt.title("Fake")
    plt.figure()
    plt.specgram(agc_d2, cmap="gray")
    #plt.title("Real")
    plt.show()
    """

    wavfile.write("phase_train_agc.wav", fs, soundsc(agc_d1))
    wavfile.write("phase_test_agc.wav", fs, soundsc(agc_d2))
    wavfile.write("phase_vq_test_agc.wav", fs, soundsc(agc_vq_d2))


def run_cqt_example():
    try:
        fs, d = fetch_sample_file("/Users/User/cqt_resources/kempff1.wav")
    except ValueError:
        print("WARNING: Using sample music instead but kempff1.wav is the example")
        fs, d = fetch_sample_music()
    X = d[:44100]
    X_cq, c_dc, c_nyq, multiscale, shift, window_lens = cqt(X, fs)
    X_r = icqt(X_cq, c_dc, c_nyq, multiscale, shift, window_lens)
    SNR = 20 * np.log10(np.linalg.norm(X - X_r) / np.linalg.norm(X))
    wavfile.write("cqt_original.wav", fs, soundsc(X))
    wavfile.write("cqt_reconstruction.wav", fs, soundsc(X_r))


def run_fft_dct_example():
    random_state = np.random.RandomState(1999)

    fs, d = fetch_sample_speech_fruit()
    n_fft = 64
    X = d[0]
    X_stft = stft(X, n_fft)
    X_rr = complex_to_real_view(X_stft)
    X_dct = fftpack.dct(X_rr, axis=-1, norm='ortho')
    X_dct_sub = X_dct[1:] - X_dct[:-1]
    std = X_dct_sub.std(axis=0, keepdims=True)
    X_dct_sub += .01 * std * random_state.randn(
        X_dct_sub.shape[0], X_dct_sub.shape[1])
    X_dct_unsub = np.cumsum(X_dct_sub, axis=0)
    X_idct = fftpack.idct(X_dct_unsub, axis=-1, norm='ortho')
    X_irr = real_to_complex_view(X_idct)
    X_r = istft(X_irr, n_fft)[:len(X)]

    SNR = 20 * np.log10(np.linalg.norm(X - X_r) / np.linalg.norm(X))
    print(SNR)

    wavfile.write("fftdct_orig.wav", fs, soundsc(X))
    wavfile.write("fftdct_rec.wav", fs, soundsc(X_r))


def run_world_example():
    fs, d = fetch_sample_speech_tapestry()
    d = d.astype("float32") / 2 ** 15
    temporal_positions_h, f0_h, vuv_h, f0_candidates_h = harvest(d, fs)
    temporal_positions_ct, spectrogram_ct, fs_ct = cheaptrick(d, fs,
            temporal_positions_h, f0_h, vuv_h)
    temporal_positions_d4c, f0_d4c, vuv_d4c, aper_d4c, coarse_aper_d4c = d4c(d, fs,
            temporal_positions_h, f0_h, vuv_h)
    #y = world_synthesis(f0_d4c, vuv_d4c, aper_d4c, spectrogram_ct, fs_ct)
    y = world_synthesis(f0_d4c, vuv_d4c, coarse_aper_d4c, spectrogram_ct, fs_ct)
    wavfile.write("out.wav", fs, soundsc(y))


def run_mgc_example():
    import matplotlib.pyplot as plt
    fs, x = wavfile.read("test16k.wav")
    pos = 3000
    fftlen = 1024
    win = np.blackman(fftlen) / np.sqrt(np.sum(np.blackman(fftlen) ** 2))
    xw = x[pos:pos + fftlen] * win
    sp = 20 * np.log10(np.abs(np.fft.rfft(xw)))
    mgc_order = 20
    mgc_alpha = 0.41
    mgc_gamma = -0.35
    mgc_arr = win2mgc(xw, order=mgc_order, alpha=mgc_alpha, gamma=mgc_gamma, verbose=True)
    xwsp = 20 * np.log10(np.abs(np.fft.rfft(xw)))
    sp = mgc2sp(mgc_arr, mgc_alpha, mgc_gamma, fftlen)
    plt.plot(xwsp)
    plt.plot(20. / np.log(10) * np.real(sp), "r")
    plt.xlim(1, len(xwsp))
    plt.show()


def run_world_mgc_example():
    fs, d = fetch_sample_speech_tapestry()
    d = d.astype("float32") / 2 ** 15

    # harcoded for 16k from
    # https://github.com/CSTR-Edinburgh/merlin/blob/master/misc/scripts/vocoder/world/extract_features_for_merlin.sh
    mgc_alpha = 0.58
    #mgc_order = 59
    mgc_order = 59
    # this is actually just mcep
    mgc_gamma = 0.0

    temporal_positions_h, f0_h, vuv_h, f0_candidates_h = harvest(d, fs)
    temporal_positions_ct, spectrogram_ct, fs_ct = cheaptrick(d, fs,
            temporal_positions_h, f0_h, vuv_h)
    temporal_positions_d4c, f0_d4c, vuv_d4c, aper_d4c, coarse_aper_d4c = d4c(d, fs,
            temporal_positions_h, f0_h, vuv_h)

    mgc_arr = sp2mgc(spectrogram_ct, mgc_order, mgc_alpha, mgc_gamma,
            verbose=True)

    from sklearn.externals import joblib
    mem = joblib.Memory("/tmp")
    mem.clear()

    sp_r = mgc2sp(mgc_arr, mgc_alpha, mgc_gamma, fs=fs, verbose=True)

    import matplotlib.pyplot as plt
    plt.imshow(20 * np.log10(sp_r))
    plt.figure()
    plt.imshow(20 * np.log10(spectrogram_ct))
    plt.show()

    y = world_synthesis(f0_d4c, vuv_d4c, coarse_aper_d4c, sp_r, fs)
    #y = world_synthesis(f0_d4c, vuv_d4c, aper_d4c, sp_r, fs)
    wavfile.write("out_mgc.wav", fs, soundsc(y))


if __name__ == "__main__":
    run_world_mgc_example()
    """
    Trying to run all examples will seg fault on my laptop - probably memory!
    Comment individually
    run_world_mgc_example()
    run_world_example()
    run_mgc_example()
    run_phase_reconstruction_example()
    run_phase_vq_example()
    run_dct_vq_example()
    run_fft_vq_example()
    run_lpc_example()
    run_cqt_example()
    run_fft_dct_example()
    test_all()
    """
