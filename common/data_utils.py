#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Data process utility functions."""
import os, sys
import numpy as np
import librosa
import sonopy

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
from classifier.params import pr


def buffer_to_audio(buffer):
    """
    convert a raw mono audio byte string to numpy array of floats
    """
    return np.fromstring(buffer, dtype='<i2').astype(np.float32, order='C') / 32768.0


def add_deltas(features):
    """
    inserts extra features that are the difference between adjacent timesteps
    """
    deltas = np.zeros_like(features)
    for i in range(1, len(features)):
        deltas[i] = features[i] - features[i - 1]

    return np.concatenate([features, deltas], -1)


def vectorize_raw(audio):
    """
    turns audio into feature vectors, without clipping for length
    """
    if len(audio) == 0:
        raise InvalidAudio('Cannot vectorize empty audio!')

    #feature = librosa.feature.mfcc(y=audio, sr=pr.sample_rate, n_mfcc=pr.n_mfcc)
    feature = sonopy.mfcc_spec(audio, pr.sample_rate, (pr.window_samples, pr.hop_samples), num_filt=pr.n_filt, fft_size=pr.n_fft, num_coeffs=pr.n_mfcc)
    return feature


def audio_to_feature(audio_data):
    """
    audio data to mfcc feature
    """
    audio_data = audio_data[:pr.max_samples]

    if len(audio_data) < pr.max_samples:
        audio_data = np.concatenate([np.zeros((pr.max_samples - len(audio_data),)), audio_data])

    feature = vectorize_raw(audio_data)
    if pr.use_delta:
        feature = add_deltas(feature)

    return feature


def get_mfcc_feature(audio_path):
    """
    convert audio sample file to mfcc feature vectors
    """
    audio_data, _ = librosa.load(audio_path, sr=pr.sample_rate, mono=True)

    feature = audio_to_feature(audio_data)

    return np.expand_dims(feature, axis=-1)

