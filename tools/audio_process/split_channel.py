#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reference from:
https://blog.csdn.net/sinat_38682860/article/details/115210524
"""
import os, sys, argparse
import glob
from tqdm import tqdm
import numpy as np
from scipy.io import wavfile
from pydub import AudioSegment


def get_channel_num(wav_file):
    sound = AudioSegment.from_file(wav_file)
    return sound.channels


def split_channel(wav_file, output_path, target_channel, clip_length):
    channel_num = get_channel_num(wav_file)
    assert channel_num > target_channel, 'split channel {} from {} channels file'.format(target_channel, channel_num)

    sample_rate, wav_data = wavfile.read(wav_file)
    channel_data = []
    for item in wav_data:
        channel_data.append(item[target_channel])

    if clip_length is not None:
        clip_samples = sample_rate * clip_length // 1000
        channel_data = channel_data[:clip_samples]

    output_file = os.path.join(output_path, os.path.splitext(os.path.basename(wav_file))[0]+'_channel{}.wav'.format(target_channel))
    wavfile.write(output_file, sample_rate, np.array(channel_data))



def main():
    parser = argparse.ArgumentParser(description='split specific channel from wav audio files')
    parser.add_argument('--wav_path', type=str, required=True,
                        help='wav file or directory to split')
    parser.add_argument('--output_path', type=str, required=True,
                        help='output path to save split audio file')
    parser.add_argument('--target_channel', type=int, required=False, default=0,
                        help='target audio channel to split. default=%(default)s')
    parser.add_argument('--clip_length', type=int, required=False, default=None,
                        help='clipped audio length in ms, None is unchange. default=%(default)s')

    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    # get wav file list or single wav audio
    if os.path.isfile(args.wav_path):
        split_channel(args.wav_path, args.output_path, args.target_channel, args.clip_length)
    else:
        wav_files = glob.glob(os.path.join(args.wav_path, '*.wav'))
        pbar = tqdm(total=len(wav_files), desc='Channel Split')

        for wav_file in wav_files:
            split_channel(wav_file, args.output_path, args.target_channel, args.clip_length)
            pbar.update(1)
        pbar.close()

    print('\nSplit finished')



if __name__ == "__main__":
    main()
