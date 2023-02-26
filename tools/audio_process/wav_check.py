#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check wav file format and collect audio length statistics
"""
import os, sys, argparse
import glob
import wave
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def show_audio_info(wf):
    print('channels: {}'.format(wf.getnchannels()))
    print('sample rate: {}'.format(wf.getframerate()))
    print('bits per sample: {}'.format(wf.getsampwidth() * 8))
    print('total frames: {}'.format(wf.getnframes()))
    print('duration seconds: {} s'.format(wf.getnframes() / wf.getframerate()))
    print('compress type: {}'.format(wf.getcomptype()))
    print('compress name: {}'.format(wf.getcompname()))


def wav_check(wav_file, channel_num, sample_rate, sample_bit):
    wf = wave.open(wav_file, 'rb')

    if wf.getnchannels() != channel_num or wf.getframerate() != sample_rate or wf.getsampwidth() * 8 != sample_bit:
        # break if any invalid audio file
        print('Invalid wav audio file:', wav_file)
        print('\nAudio info:')
        show_audio_info(wf)
        exit()

    wav_length = (wf.getnframes() / wf.getframerate())
    wf.close()

    return wav_length



def main():
    parser = argparse.ArgumentParser(description='check wav file format and collect audio length statistics')
    parser.add_argument('--wav_path', type=str, required=True,
                        help='wav file or directory to check')
    parser.add_argument('--channel_num', type=int, required=False, default=1,
                        help='target channel number. default=%(default)s')
    parser.add_argument('--sample_rate', type=int, required=False, default=16000, choices=[None, 8000, 16000, 22050, 44100, 48000],
                        help='target sample rate. default=%(default)s')
    parser.add_argument('--sample_bit', type=int, required=False, default=16, choices=[None, 8, 16, 24, 32],
                        help='target sample bit number. default=%(default)s')
    parser.add_argument('--length_threshold', type=float, required=False, default=1.5,
                        help='audio length threshold in second. default=%(default)s')

    args = parser.parse_args()

    # get wav audio file list or single audio
    if os.path.isfile(args.wav_path):
        wf = wave.open(args.wav_path, 'rb')
        print('\nAudio file info:')
        show_audio_info(wf)
        wf.close()
    else:
        wav_files = glob.glob(os.path.join(args.wav_path, '*.wav'))

        # init audio length statistics
        above_threshold_num = 0
        length_list = []

        pbar = tqdm(total=len(wav_files), desc='wav format check')
        for wav_file in wav_files:
            wav_length = wav_check(wav_file, args.channel_num, args.sample_rate, args.sample_bit)
            # update statistics
            length_list.append(wav_length)

            if wav_length > args.length_threshold:
                above_threshold_num += 1

            pbar.update(1)
        pbar.close()
        length_array = np.array(length_list)

        print('Format check success\n')
        print('Total audio number: {}'.format(len(wav_files)))
        print('Max audio length: {} s'.format(length_array.max()))
        print('Min audio length: {} s'.format(length_array.min()))
        print('Average audio length: {} s'.format(length_array.mean()))
        print('Total audio length: {} s'.format(length_array.sum()))
        print('Audio number above {} s: {}'.format(args.length_threshold, above_threshold_num))

        # show hist for wav audio length distribution
        plt.hist(length_array, bins=40, alpha=0.7)
        plt.xlabel("length(second)")
        plt.ylabel("number")
        plt.title("hist for length distribution of {} wav files".format(len(wav_files)))
        plt.show()
    print('\nDone')


if __name__ == "__main__":
    main()
