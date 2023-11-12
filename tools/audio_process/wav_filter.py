#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apply butterworth filter to wav audio file
Reference from:
https://www.cnblogs.com/xiaosongshine/p/10831931.html
"""
import os, sys, argparse
import numpy as np
from scipy import signal
import wavio


def load_audio(audio_file):
    wav = wavio.read(audio_file)

    if wav.data.dtype != np.int16:
        raise ValueError('Unsupported data type: ' + str(wav.data.dtype))

    data = np.squeeze(wav.data)
    norm_data = data.astype(np.float32) / float(np.iinfo(data.dtype).max)

    return norm_data, wav.rate


def save_audio(filename, audio, sample_rate):
    """
    Save loaded audio to file using the configured audio parameters
    """
    save_audio = (audio * np.iinfo(np.int16).max).astype(np.int16)
    #wavio.write(filename, save_audio, sample_rate, sampwidth=2, scale='none')
    wavio.write(filename, save_audio, sample_rate, sampwidth=2)


def wav_filter(wav_file, filter_type, filter_order, up_limit_freq, down_limit_freq, output_file):
    audio_data, sample_rate = load_audio(wav_file)

    print('sample rate of wav:', sample_rate)

    if filter_type == 'lowpass':
        wn = 2 * up_limit_freq / sample_rate
        b, a = signal.butter(filter_order, wn, 'lowpass')
    elif filter_type == 'highpass':
        wn = 2 * down_limit_freq / sample_rate
        b, a = signal.butter(filter_order, wn, 'highpass')
    elif filter_type == 'bandpass':
        wn_up = 2 * up_limit_freq / sample_rate
        wn_down = 2 * down_limit_freq / sample_rate
        b, a = signal.butter(filter_order, [wn_down, wn_up], 'bandpass')
    elif filter_type == 'bandstop':
        wn_up = 2 * up_limit_freq / sample_rate
        wn_down = 2 * down_limit_freq / sample_rate
        b, a = signal.butter(filter_order, [wn_down, wn_up], 'bandstop')
    else:
        raise ValueError('Invalid filter type')

    filted_data = signal.filtfilt(b, a, audio_data).astype(np.float32)

    # save filtered audio
    save_audio(output_file, filted_data, sample_rate)
    print('filtered audio saved to:', output_file)


def main():
    parser = argparse.ArgumentParser(description='apply butterworth filter to wav audio file. only support 16 bit format')
    parser.add_argument('--wav_file', type=str, required=True,
                        help='wav audio file')
    parser.add_argument('--filter_type', type=str, required=False, default='highpass', choices=['lowpass', 'highpass', 'bandpass', 'bandstop'],
                        help='audio filter type. default=%(default)s')
    parser.add_argument('--filter_order', type=int, required=False, default=4,
                        help='order of the filter. default=%(default)s')
    parser.add_argument('--up_limit_freq', type=int, required=False, default=None,
                        help='up limit frequency for filter. default=%(default)s')
    parser.add_argument('--down_limit_freq', type=int, required=False, default=None,
                        help='down limit frequency for filter. default=%(default)s')
    parser.add_argument('--output_file', type=str, required=True,
                        help='output filtered wav audio file')

    args = parser.parse_args()

    wav_filter(args.wav_file, args.filter_type, args.filter_order, args.up_limit_freq, args.down_limit_freq, args.output_file)



if __name__ == "__main__":
    main()
