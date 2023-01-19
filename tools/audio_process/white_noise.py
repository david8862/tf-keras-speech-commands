#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, argparse
import numpy as np
from scipy import stats
from pydub import AudioSegment


def generate_white_noise(length, sample_rate, sample_bit=16, amplitude=0.7):
    amplitude_bit = int(sample_bit * amplitude)
    assert (sample_bit == 16), 'only support 16 bit sample for white noise generate'

    noise_data = stats.truncnorm(-1, 1, scale=min(2**sample_bit, 2**amplitude_bit)).rvs(int(sample_rate * (length / 1000.0)))
    noise_data = noise_data.astype(np.int16)
    return noise_data


def white_noise(length, sample_rate, amplitude, output_file):
    noise_data = generate_white_noise(length, sample_rate, sample_bit=16, amplitude=amplitude)
    noise_audio = AudioSegment(noise_data.tobytes(),
                               frame_rate=sample_rate,
                               sample_width=noise_data.dtype.itemsize,
                               channels=1)

    noise_audio.export(output_file, format='wav')


def main():
    parser = argparse.ArgumentParser(description='generate white noise .wav audio file. only support 16 bit format')
    parser.add_argument('--length', type=int, required=False, default=1000,
                        help='target noise audio length in ms. default=%(default)s')
    parser.add_argument('--sample_rate', type=int, required=False, default=16000, choices=[8000, 16000, 22050, 44100, 48000],
                        help='audio sample rate. default=%(default)s')
    parser.add_argument('--amplitude', type=float, required=False, default=0.7,
                        help='white noise amplitude. default=%(default)s')
    parser.add_argument('--output_file', type=str, required=True,
                        help='merged audio file')

    args = parser.parse_args()


    white_noise(args.length, args.sample_rate, args.amplitude, args.output_file)



if __name__ == "__main__":
    main()
