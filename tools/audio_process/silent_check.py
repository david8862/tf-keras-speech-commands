#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check & filter silent audio file with average energy
"""
import os, sys, argparse
import glob
import shutil
import numpy as np
import librosa
from tqdm import tqdm


def silent_check(wav_file, threshold):
    audio, sr = librosa.load(wav_file, sr=None)

    energy = np.sum(abs(audio**2))
    energy_per_second = energy / (len(audio) / sr)
    #print('energy_per_second:', energy_per_second)

    if energy_per_second < threshold:
        return True
    else:
        return False



def main():
    parser = argparse.ArgumentParser(description='check & filter silent audio file with average energy')
    parser.add_argument('--wav_path', type=str, required=True,
                        help='wav file or directory to check')
    parser.add_argument('--threshold', type=float, required=False, default=0.2,
                        help='threshold of energy per second to check if is a silent audio. default=%(default)s')
    parser.add_argument('--target_path', type=str, required=True,
                        help='target path to save silent audio files')

    args = parser.parse_args()

    os.makedirs(args.target_path, exist_ok=True)

    # get wav audio file list or single audio
    if os.path.isfile(args.wav_path):
        is_silent = silent_check(args.wav_path, args.threshold)
        print('silent flag for {}: {}'.format(args.wav_path, is_silent))
    else:
        wav_files = glob.glob(os.path.join(args.wav_path, '*.wav'))
        silent_count = 0

        pbar = tqdm(total=len(wav_files), desc='speech duration check')
        for wav_file in wav_files:
            is_silent = silent_check(wav_file, args.threshold)
            if is_silent:
                shutil.move(wav_file, args.target_path)
                silent_count += 1

            pbar.update(1)
        pbar.close()
        print('\nFound {} silent audio files'.format(silent_count))

    print('\nDone')


if __name__ == "__main__":
    main()
