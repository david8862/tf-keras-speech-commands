#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parameters used in the audio pipeline
These configure the following stages:
 - Conversion from audio to input vectors
 - Interpretation of the network output to a confidence value
"""
import os
from math import floor

import attr
import json


@attr.s(frozen=True)
class ListenerParams:
    """
    General pipeline information:
     - Audio goes through a series of transformations to convert raw audio into machine readable data
     - These transformations are as follows:
       - Raw audio -> chopped audio
         - buffer_t, sample_depth: Input audio loaded and truncated using these value
         - window_t, hop_t: Linear audio chopped into overlapping frames using a sliding window
       - Chopped audio -> FFT spectrogram
         - n_fft, sample_rate: Each audio frame is converted to n_fft frequency intensities
       - FFT spectrogram -> Mel spectrogram (compressed)
         - n_filt: Each fft frame is compressed to n_filt summarized mel frequency bins/bands
       - Mel spectrogram -> MFCC
         - n_mfcc: Each mel frame is converted to MFCCs and the first n_mfcc values are taken
       - Disabled by default: Last phase -> Delta vectors
         - use_delta: If this value is true, the difference between consecutive vectors is concatenated to each frame

    Parameters for audio pipeline:
     - buffer_t: Input size of audio. Wakeword must fit within this time
     - window_t: Time of the window used to calculate a single spectrogram frame
     - hop_t: Time the window advances forward to calculate the next spectrogram frame
     - sample_rate: Input audio sample rate
     - sample_depth: Bytes per input audio sample
     - n_fft: Size of FFT to generate from audio frame
     - n_filt: Number of filters to compress FFT to
     - n_mfcc: Number of MFCC coefficients to use
     - use_delta: If True, generates "delta vectors" before sending to network
     - threshold_config: Output distribution configuration automatically generated from precise-calc-threshold
     - threshold_center: Output distribution center automatically generated from precise-calc-threshold
    """
    buffer_t = attr.ib()  # type: float
    window_t = attr.ib()  # type: float
    hop_t = attr.ib()  # type: float
    sample_rate = attr.ib()  # type: int
    sample_depth = attr.ib()  # type: int
    n_fft = attr.ib()  # type: int
    n_filt = attr.ib()  # type: int
    n_mfcc = attr.ib()  # type: int
    use_delta = attr.ib()  # type: bool
    threshold_config = attr.ib()  # type: tuple
    threshold_center = attr.ib()  # type: float

    @property
    def buffer_samples(self):
        """buffer_t converted to samples, truncating partial frames"""
        samples = int(self.sample_rate * self.buffer_t + 0.5)
        return self.hop_samples * (samples // self.hop_samples)

    @property
    def n_features(self):
        """Number of timesteps in one input to the network"""
        return 1 + int(floor((self.buffer_samples - self.window_samples) / self.hop_samples))

    @property
    def window_samples(self):
        """window_t converted to samples"""
        return int(self.sample_rate * self.window_t + 0.5)

    @property
    def hop_samples(self):
        """hop_t converted to samples"""
        return int(self.sample_rate * self.hop_t + 0.5)

    @property
    def max_samples(self):
        """The input size converted to audio samples"""
        return int(self.buffer_t * self.sample_rate)

    @property
    def feature_size(self):
        """The size of an input vector generated with these parameters"""
        num_features = self.n_mfcc
        if self.use_delta:
            num_features *= 2
        return num_features


# Global listener parameters
# These are the default values for all parameters
# These were selected tentatively to balance CPU usage with accuracy
# For the Hey Mycroft wake word, small changes to these parameters
# did not make a significant difference in accuracy
pr = ListenerParams(
    buffer_t=1.0, window_t=0.064, hop_t=0.032, sample_rate=16000,
    sample_depth=2, n_fft=1024, n_filt=20, n_mfcc=20, use_delta=False,
    threshold_config=((6, 4),), threshold_center=0.2
)



def inject_params(params_file):
    """Set the global listener params to a saved model"""
    try:
        with open(params_file) as f:
            pr.__dict__.update(**json.load(f))
    except (OSError, ValueError, TypeError):
        if os.path.isfile(params_file):
            print('Warning: Failed to load parameters from ' + params_file)
    return pr


def save_params(params_file):
    """Save current global listener params to a file"""
    with open(params_file, 'w') as f:
        json.dump(pr.__dict__, f, indent=2)
