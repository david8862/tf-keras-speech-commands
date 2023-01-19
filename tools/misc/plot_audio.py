#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot waveform & spectrogram for audio sample files

Reference from:
https://www.cnblogs.com/LXP-Never/p/13404523.html
"""
import os, sys, argparse
import glob
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import librosa


def plot_audio(audio_file, sample_rate, output_path):
    wav_data, _ = librosa.load(audio_file, sr=sample_rate, mono=True)

    # draw waveform
    plt.subplot(2, 1, 1)
    plt.title("Waveform", fontsize='large')
    time = np.arange(0, len(wav_data)) * (1.0 / sample_rate)
    plt.plot(time, wav_data)
    plt.xlabel('Time/s', fontsize='large')
    plt.ylabel('Amplitude', fontsize='large')

    # draw spectrogram
    plt.subplot(2, 1, 2)
    plt.title("Spectrogram", fontsize='large')
    plt.specgram(wav_data, Fs=sample_rate, scale_by_freq=True, sides='default', cmap="jet")
    plt.xlabel('Time/s', fontsize='large')
    plt.ylabel('Freq/Hz', fontsize='large')

    # save or show result
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.splitext(os.path.basename(audio_file))[0]
        output_file = os.path.join(output_path, output_file+'.jpg')
        plt.savefig(output_file, dpi=75)
    else:
        plt.tight_layout()
        plt.show()

    return



def main():
    parser = argparse.ArgumentParser(description='Plot waveform & spectrogram for audio file')
    parser.add_argument('--audio_path', type=str, required=True,
                        help='audio file or directory to plot')
    parser.add_argument('--sample_rate', type=int, required=False, default=16000, choices=[8000, 16000, 22050, 44100, 48000],
                        help='audio sample rate. default=%(default)s')
    parser.add_argument('--output_path', type=str, required=False, default=None,
                        help='output path to save chart, default=%(default)s')

    args = parser.parse_args()

    # get audio file list or single audio
    if os.path.isfile(args.audio_path):
        plot_audio(args.audio_path, args.sample_rate, args.output_path)
    else:
        audio_files = glob.glob(os.path.join(args.audio_path, '*'))
        pbar = tqdm(total=len(audio_files), desc='Plot spectrogram')

        for audio_file in audio_files:
            plot_audio(audio_file, args.sample_rate, args.output_path)
            pbar.update(1)
        pbar.close()


if __name__ == "__main__":
    main()
