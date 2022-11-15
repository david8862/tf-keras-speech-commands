#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot Mel/MFCC spectrogram for audio sample files

Reference from:
https://blog.csdn.net/weixin_50547200/article/details/117294164
"""
import os, sys, argparse
import glob
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import librosa
from sonopy import mfcc_spec, mel_spec

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
from classifier.params import pr, inject_params


def plot_spectrogram(audio_file, spec_type, output_path):
    data, _ = librosa.load(audio_file, sr=pr.sample_rate, mono=True)

    if spec_type == 'mel':
        spec = mel_spec(data, pr.sample_rate, (pr.window_samples, pr.hop_samples), num_filt=pr.n_filt, fft_size=pr.n_fft)
        coef_len = pr.n_filt
    elif spec_type == 'mfcc':
        spec = mfcc_spec(data, pr.sample_rate, (pr.window_samples, pr.hop_samples), num_filt=pr.n_filt, fft_size=pr.n_fft, num_coeffs=pr.n_mfcc)
        coef_len = pr.n_mfcc

    tick_num = 5
    ticks_coef = np.arange(0, coef_len, coef_len/tick_num, dtype=np.float32)
    ticks_freq = np.arange(0, pr.sample_rate, pr.sample_rate/tick_num, dtype=np.int32)

    plt.imshow(spec.T, cmap=plt.cm.jet, aspect='auto')
    plt.yticks(ticks_coef, ticks_freq)

    # set chart title & label
    plt.title('%s spectrogram' % spec_type, fontsize='large')
    plt.xlabel('frame', fontsize='large')
    plt.ylabel('frequency', fontsize='large')

    # save or show result
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.splitext(os.path.basename(audio_file))[0]
        output_file = os.path.join(output_path, output_file+'.jpg')
        plt.savefig(output_file, dpi=75)
    else:
        plt.show()

    return


def main():
    parser = argparse.ArgumentParser(description='Plot Mel/MFCC feature spectrogram for audio file')
    parser.add_argument('--audio_path', type=str, required=True,
                        help='audio file or directory to plot')
    parser.add_argument('--params_path', type=str, required=False, default=None,
                        help='path to params json file')
    parser.add_argument('--spec_type', type=str, required=False, default='mel', choices=['mel', 'mfcc'],
                        help='spectrogram type to plot (mel/mfcc), default=%(default)s')
    parser.add_argument('--output_path', type=str, required=False, default=None,
                        help='output path to save spectrogram, default=%(default)s')

    args = parser.parse_args()

    # load & update audio params
    if args.params_path:
        inject_params(args.params_path)

    # get audio file list or single audio
    if os.path.isfile(args.audio_path):
        plot_spectrogram(args.audio_path, args.spec_type, args.output_path)
    else:
        audio_files = glob.glob(os.path.join(args.audio_path, '*'))
        pbar = tqdm(total=len(audio_files), desc='Plot spectrogram')

        for audio_file in audio_files:
            plot_spectrogram(audio_file, args.spec_type, args.output_path)
            pbar.update(1)
        pbar.close()


if __name__ == "__main__":
    main()
