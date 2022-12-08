#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check MFCC feature array for input wav file
"""
import os, sys, argparse
import numpy as np
import wave, wavio
from math import floor

import sonopy
import speechpy
import librosa
import python_speech_features


def load_audio(audio_file, sample_rate):
    wav = wavio.read(audio_file)

    if wav.rate != sample_rate:
        raise ValueError('Unsupported sample rate: ' + str(wav.rate))
    if wav.data.dtype != np.int16:
        raise ValueError('Unsupported data type: ' + str(wav.data.dtype))

    data = np.squeeze(wav.data)
    return data.astype(np.float32) / float(np.iinfo(data.dtype).max)



def mfcc_feature(wav_file, package_type, output_file, sample_rate, buffer_t, window_t, hop_t, n_fft, n_filt, n_mfcc):
    # load audio data
    audio_data = load_audio(wav_file, sample_rate)

    max_samples = int(buffer_t * sample_rate)
    window_samples = int(sample_rate * window_t + 0.5)
    hop_samples = int(sample_rate * hop_t + 0.5)

    if len(audio_data) > max_samples:
        audio_data = audio_data[-max_samples:]

    n_features = 1 + int(floor((max_samples - window_samples) / hop_samples))

    if package_type == 'sonopy':
        features = sonopy.mfcc_spec(audio_data, sample_rate, (window_samples, hop_samples), num_filt=n_filt, fft_size=n_fft, num_coeffs=n_mfcc)
    elif package_type == 'speechpy':
        features = speechpy.feature.mfcc(audio_data, sample_rate, window_t, hop_t, n_mfcc, n_filt, n_fft)
    elif package_type == 'librosa':
        features = librosa.feature.mfcc(y=audio_data, sr=sample_rate, S=None, n_mfcc=n_mfcc, dct_type=2, norm='ortho', n_fft=n_fft, hop_length=hop_samples).transpose()
    elif package_type == 'python_speech_features':
        # seems python_speech_features use raw audio samples as input,
        # so we convert it back
        audio_data *= float(np.iinfo(np.int16).max)
        audio_data = audio_data.astype(np.int16)
        features = python_speech_features.mfcc(audio_data, samplerate=sample_rate, winlen=window_t, winstep=hop_t, numcep=n_mfcc, nfilt=n_filt, nfft=n_fft, lowfreq=300, highfreq=sample_rate//2)

    if len(features) < n_features:
        features = np.concatenate([
            np.zeros((n_features - len(features), features.shape[1])),
            features
        ])
    if len(features) > n_features:
        features = features[-n_features:]

    np.savetxt(output_file, features)
    print('output MFCC feature shape:', features.shape)
    print('feature array has been saved to ', output_file)



def main():
    parser = argparse.ArgumentParser(description='generate mfcc feature array for input wav file')
    parser.add_argument('--wav_path', type=str, required=True,
                        help='wav file or directory for input audio')
    parser.add_argument('--package_type', type=str, required=False, default='sonopy', choices=['sonopy', 'speechpy', 'librosa', 'python_speech_features'],
                        help='python package for mfcc feature extraction. default=%(default)s')
    parser.add_argument('--output_file', type=str, required=True,
                        help='output txt file to save mfcc feature array')

    parser.add_argument('--sample_rate', type=int, required=False, default=16000, choices=[None, 8000, 16000, 22050, 44100, 48000],
                        help='needed sample rate. default=%(default)s')
    parser.add_argument('--buffer_t', type=float, required=False, default=1.5,
                        help='audio buffer size in second. default=%(default)s')
    parser.add_argument('--window_t', type=float, required=False, default=0.1,
                        help='frame window size in second. default=%(default)s')
    parser.add_argument('--hop_t', type=float, required=False, default=0.05,
                        help='window sliding stride in second. default=%(default)s')
    parser.add_argument('--n_fft', type=int, required=False, default=512,
                        help='FFT size. default=%(default)s')
    parser.add_argument('--n_filt', type=int, required=False, default=20,
                        help='mel filterbanks number. default=%(default)s')
    parser.add_argument('--n_mfcc', type=int, required=False, default=13,
                        help='mfcc coefficient number. default=%(default)s')

    args = parser.parse_args()

    mfcc_feature(args.wav_path, args.package_type, args.output_file, args.sample_rate, args.buffer_t, args.window_t, args.hop_t, args.n_fft, args.n_filt, args.n_mfcc)



if __name__ == "__main__":
    main()
