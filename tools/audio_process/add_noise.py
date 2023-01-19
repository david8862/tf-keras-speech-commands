#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
randomly add background noise to voice audio, with specified SNR

Reference from:
https://www.cnblogs.com/LXP-Never/p/13404523.html
"""
import os, sys, argparse
import glob
from tqdm import tqdm
from random import random, choice
from shutil import copy
import numpy as np
import soundfile
import librosa


def add_noise(voice_file, noise_file, snr, sample_rate, output_path):
    voice_data, _ = librosa.load(voice_file, sr=sample_rate, mono=True)
    noise_data, _ = librosa.load(noise_file, sr=sample_rate, mono=True)

    # use shorter one as audio length
    length = len(voice_data) if len(voice_data) < len(noise_data) else len(noise_data)

    voice_data = voice_data[:length]
    noise_data = noise_data[:length]

    p_voice = np.mean(voice_data ** 2)  # power of clean voice
    p_noise = np.mean(noise_data ** 2)  # power of noise

    scalar = np.sqrt(p_voice / (10 ** (snr / 10)) / (p_noise + np.finfo(np.float32).eps))
    merge_data = voice_data + scalar * noise_data

    # save merged audio
    output_file = os.path.join(output_path, os.path.splitext(os.path.basename(voice_file))[0]+'_noised.wav')
    soundfile.write(output_file, merge_data, sample_rate)



def main():
    parser = argparse.ArgumentParser(description='randomly add background noise to voice audio, with specified SNR')
    parser.add_argument('--voice_path', type=str, required=True,
                        help='voice audio file or directory')
    parser.add_argument('--noise_path', type=str, required=True,
                        help='background noise audio file or directory')
    parser.add_argument('--snr', type=str, required=False, default='50',
                        help='Sound Noise Ratio (SNR) choice in dB, separate with comma if more than one. default=%(default)s')
    parser.add_argument('--noised_rate', type=float, required=False, default=1.0,
                        help='random percentage rate of adding noise to voice audio (0.0~1.0). default=%(default)s')
    parser.add_argument('--sample_rate', type=int, required=False, default=16000, choices=[8000, 16000, 22050, 44100, 48000],
                        help='audio sample rate. default=%(default)s')
    parser.add_argument('--output_path', type=str, required=True,
                        help='output path to save merged audio file')

    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    # get noise audio file list or single noise audio
    if os.path.isdir(args.noise_path):
        noise_files = glob.glob(os.path.join(args.noise_path, '*.wav'))
    else:
        noise_files = [args.noise_path]

    # parse SNR to list
    snr_list = [int(snr) for snr in args.snr.split(',')]

    # process single voice file, or loop for voice file list
    if os.path.isfile(args.voice_path):
        # random pick noise file & SNR from list
        noise_file = choice(noise_files)
        snr = choice(snr_list)

        add_noise(args.voice_path, noise_file, snr, args.sample_rate, args.output_path)
    else:
        voice_files = glob.glob(os.path.join(args.voice_path, '*.wav'))
        pbar = tqdm(total=len(voice_files), desc='Adding Noise')
        noised_count = 0

        for voice_file in voice_files:
            # random pick noise file & SNR from list
            noise_file = choice(noise_files)
            snr = choice(snr_list)

            # add noise to voice, or just copy voice file directly
            if random() <= args.noised_rate:
                add_noise(voice_file, noise_file, snr, args.sample_rate, args.output_path)
                noised_count += 1
            else:
                copy(voice_file, args.output_path)

            pbar.update(1)
        pbar.close()
        print('Noised audio: {}/{}'.format(noised_count, len(voice_files)))
    print('Done.')


if __name__ == "__main__":
    main()
