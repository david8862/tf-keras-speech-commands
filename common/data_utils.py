#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Data process utility functions."""
import os, sys
import numpy as np
import librosa
import sonopy

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
from classifier.params import pr


def add_deltas(features):
    """
    inserts extra features that are the difference between adjacent timesteps
    """
    deltas = np.zeros_like(features)
    for i in range(1, len(features)):
        deltas[i] = features[i] - features[i - 1]

    return np.concatenate([features, deltas], -1)


def get_mfcc_feature(audio_path):
    """
    convert audio data to mfcc feature vectors
    """
    audio_data, _ = librosa.load(audio_path, sr=pr.sample_rate, mono=True)
    audio_data = audio_data[:pr.max_samples]

    if len(audio_data) < pr.max_samples:
        audio_data = np.concatenate([np.zeros((pr.max_samples - len(audio_data),)), audio_data])

    #feature = librosa.feature.mfcc(y=audio_data, sr=pr.sample_rate, n_mfcc=pr.n_mfcc)
    feature = sonopy.mfcc_spec(audio_data, pr.sample_rate, (pr.window_samples, pr.hop_samples), num_filt=pr.n_filt, fft_size=pr.n_fft, num_coeffs=pr.n_mfcc)

    if pr.use_delta:
        feature = add_deltas(feature)

    return np.expand_dims(feature, axis=-1)

