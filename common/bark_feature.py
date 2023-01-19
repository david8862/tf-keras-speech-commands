#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calculate BARK feature for an audio signal

Reference from:
https://www.cnblogs.com/LXP-Never/p/16011229.html
https://github.com/SuperKogito/spafe
"""
import numpy as np
from functools import lru_cache
from scipy.fftpack import dct
import librosa


def hz2bark_1961(Hz):
    return 13.0 * np.arctan(0.00076 * Hz) + 3.5 * np.arctan((Hz / 7500.0) ** 2)

def hz2bark_1990(Hz):
    bark_scale = (26.81 * Hz) / (1960 + Hz) - 0.5
    return bark_scale

def hz2bark_1992(Hz):
    return 6 * np.arcsinh(Hz / 600)


def hz2bark(f):
    """ Hz to bark频率 (Wang, Sekey & Gersho, 1992.) """
    return 6. * np.arcsinh(f / 600.)


def bark2hz(fb):
    """ Bark频率 to Hz """
    return 600. * np.sinh(fb / 6.)


def fft2hz(fft, sample_rate=16000, nfft=512):
    """ FFT频点 to Hz """
    return (fft * sample_rate) / (nfft + 1)


def hz2fft(fb, sample_rate=16000, nfft=512):
    """ Bark频率 to FFT频点 """
    return (nfft + 1) * fb / sample_rate


def fft2bark(fft, sample_rate=16000, nfft=512):
    """ FFT频点 to Bark频率 """
    return hz2bark((fft * sample_rate) / (nfft + 1))


def bark2fft(fb, sample_rate=16000, nfft=512):
    """ Bark频率 to FFT频点 """
    # bin = sample_rate/2 / nfft/2=sample_rate/nfft    # 每个频点的频率数
    # bins = hz_points/bin=hz_points*nfft/ sample_rate    # hz_points对应第几个fft频点
    return (nfft + 1) * bark2hz(fb) / sample_rate


def Fm(fb, fc):
    """ 计算一个特定的中心频率的Bark filter
    :param fb: frequency in Bark.
    :param fc: center frequency in Bark.
    :return: 相关的Bark filter 值/幅度
    """
    if fc - 2.5 <= fb <= fc - 0.5:
        return 10 ** (2.5 * (fb - fc + 0.5))
    elif fc - 0.5 < fb < fc + 0.5:
        return 1
    elif fc + 0.5 <= fb <= fc + 1.3:
        return 10 ** (-2.5 * (fb - fc - 0.5))
    else:
        return 0


def safe_log(x):
    """Prevents error on log(0) or log(-1)"""
    return np.log(np.clip(x, np.finfo(float).eps, None))


def chop_array(arr, window_size, hop_size):
    """chop_array([1,2,3], 2, 1) -> [[1,2], [2,3]]"""
    return [arr[i - window_size:i] for i in range(window_size, len(arr) + 1, hop_size)]


def power_spec(audio: np.ndarray, window_stride=(160, 80), fft_size=512):
    """Calculates power spectrogram"""
    frames = chop_array(audio, *window_stride) or np.empty((0, window_stride[0]))
    fft = np.fft.rfft(frames, n=fft_size)
    return (fft.real ** 2 + fft.imag ** 2) / fft_size


@lru_cache()  # Prevents recalculating when calling with same parameters
def bark_filterbanks(nfilts=20, nfft=512, sample_rate=16000, low_freq=0, high_freq=None, scale="constant"):
    """ 计算Bark-filterbanks,(B,F)
    :param nfilts: 滤波器组中滤波器的数量 (Default 20)
    :param nfft: FFT size.(Default is 512)
    :param sample_rate: 采样率，(Default 16000 Hz)
    :param low_freq: MEL滤波器的最低带边。(Default 0 Hz)
    :param high_freq: MEL滤波器的最高带边。(Default samplerate/2)
    :param scale (str): 选择Max bins 幅度 "ascend"(上升)，"descend"(下降)或 "constant"(恒定)(=1)。默认是"constant"
    :return:一个大小为(nfilts, nfft/2 + 1)的numpy数组，包含滤波器组。
    """
    # init freqs
    high_freq = high_freq or sample_rate / 2
    low_freq = low_freq or 0

    # 按Bark scale 均匀间隔计算点数(点数以Bark为单位)
    low_bark = hz2bark(low_freq)
    high_bark = hz2bark(high_freq)
    bark_points = np.linspace(low_bark, high_bark, nfilts + 4)

    bins = np.floor(bark2fft(bark_points))  # Bark Scale等分布对应的 FFT bin number
    # [  0.   2.   5.   7.  10.  13.  16.  20.  24.  28.  33.  38.  44.  51.
    #   59.  67.  77.  88. 101. 115. 132. 151. 172. 197. 224. 256.]
    fbank = np.zeros([nfilts, nfft // 2 + 1])

    # init scaler
    if scale == "descendant" or scale == "constant":
        c = 1
    else:
        c = 0

    for i in range(0, nfilts):      # --> B
        # compute scaler
        if scale == "descendant":
            c -= 1 / nfilts
            c = c * (c > 0) + 0 * (c < 0)
        elif scale == "ascendant":
            c += 1 / nfilts
            c = c * (c < 1) + 1 * (c > 1)

        for j in range(int(bins[i]), int(bins[i + 4])):     # --> F
            fc = bark_points[i+2]   # 中心频率
            fb = fft2bark(j)        # Bark 频率
            fbank[i, j] = c * Fm(fb, fc)
    return np.abs(fbank)


def bark_spec(audio, sample_rate, window_size, hop_size, fft_size=512, num_filt=24):
    # use librosa.stft or implement from sonopy to calculate power spec
    powers = power_spec(audio, (window_size, hop_size), fft_size)
    #S = librosa.stft(audio, n_fft=fft_size, hop_length=hop_size, win_length=window_size, window="hann", center=False)
    #powers = np.abs(S).T  # power spec, refer librosa.magphase()

    # get bark filterbanks
    filterbanks = bark_filterbanks(nfilts=num_filt, nfft=fft_size, sample_rate=sample_rate, low_freq=0, high_freq=None, scale="constant")

    # bark power spectrogram (log scale)
    barks = np.dot(powers, filterbanks.T)
    #barks = 20 * np.log10(barks)  # dB
    barks = safe_log(barks)

    return barks


def bfcc_spec(audio, sample_rate, window_size, hop_size, fft_size=512, num_filt=26, num_coeffs=13):
    # use librosa.stft or implement from sonopy to calculate power spec
    powers = power_spec(audio, (window_size, hop_size), fft_size)
    #S = librosa.stft(audio, n_fft=fft_size, hop_length=hop_size, win_length=window_size, window="hann", center=False)
    #powers = np.abs(S).T  # power spec, refer librosa.magphase()
    if powers.size == 0:
        return np.empty((0, min(num_filt, num_coeffs)))

    # get bark filterbanks
    filterbanks = bark_filterbanks(nfilts=num_filt, nfft=fft_size, sample_rate=sample_rate, low_freq=0, high_freq=None, scale="constant")

    # bark power spectrogram (log scale)
    barks = np.dot(powers, filterbanks.T)
    #barks = 20 * np.log10(barks)  # dB
    barks = safe_log(barks)

    bfccs = dct(barks, norm='ortho')[:, :num_coeffs]  # machine readable spectrogram
    bfccs[:, 0] = safe_log(np.sum(powers, 1))  # Replace first band with log energies

    return bfccs



if __name__ == "__main__":
    import librosa.display
    import matplotlib.pyplot as plt
    nfilts = 22
    fft_size = 512
    sample_rate = 16000

    filterbanks = bark_filterbanks(nfilts=nfilts, nfft=fft_size, sample_rate=sample_rate, low_freq=0, high_freq=None, scale="constant")
    # ================ draw bark filterbanks ===========================
    fft_len = fft_size // 2 + 1
    freq_bin = sample_rate // 2 / (fft_size // 2)  # 一个频点多少Hz
    x = np.linspace(0, fft_len, fft_len)

    plt.plot(x * freq_bin, filterbanks.T)
    plt.title('Bark Filterbanks', fontsize=14)
    plt.tight_layout()
    plt.grid(linestyle='--')
    plt.show()

    wav, _ = librosa.load("sheila_2.wav", sr=sample_rate)
    barks = bark_spec(wav, sample_rate, window_size=fft_size, hop_size=fft_size//2, fft_size=fft_size, num_filt=nfilts)
    # ================ draw bark spectrogram ==========================
    plt.figure()
    librosa.display.specshow(barks.T, sr=sample_rate, x_axis='time', y_axis='linear', cmap="jet")
    plt.title('Bark Spectrogram', fontsize=14)
    plt.xlabel('Time/s', fontsize=14)
    plt.ylabel('Freq/kHz', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.tight_layout()
    plt.show()


    num_coeffs = 13
    bfccs = bfcc_spec(wav, sample_rate, window_size=fft_size, hop_size=fft_size//2, fft_size=fft_size, num_filt=nfilts, num_coeffs=num_coeffs)
    # ================ draw bfcc spectrogram ==========================
    plt.figure()
    librosa.display.specshow(bfccs.T, sr=sample_rate, x_axis='time', y_axis='linear', cmap="jet")
    plt.title('BFCC Spectrogram', fontsize=14)
    plt.xlabel('Time/s', fontsize=14)
    plt.ylabel('Freq/kHz', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.tight_layout()
    plt.show()

