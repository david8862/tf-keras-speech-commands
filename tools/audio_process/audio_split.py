#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reference from:
https://blog.csdn.net/Songyongchao1995/article/details/110249948
"""
import os, sys, argparse
import glob
from tqdm import tqdm
import librosa
import soundfile as sf
from pydub import AudioSegment
from pydub.utils import make_chunks
#import numpy as np


def audio_resample(audio_file, output_file, sample_rate):
    sound, sr = librosa.load(audio_file, sr=None)
    sound_resample = librosa.resample(sound, orig_sr=sr, target_sr=sample_rate)
    sf.write(output_file, sound_resample, sample_rate)


def show_audio_info(sound):
    print('channels: {}'.format(sound.channels))
    print('duration seconds: {} s'.format(sound.duration_seconds))
    print('loudness: {} dB'.format(sound.dBFS))
    print('max loudness: {} dB'.format(sound.max_dBFS))
    print('raw loudness: {}'.format(sound.rms))
    print('raw max loudness: {}'.format(sound.max))
    print('sample rate: {}'.format(sound.frame_rate))
    print('total frames: {}'.format(sound.frame_count()))
    print('bits per frame: {}'.format(sound.frame_width * 8))
    print('bits per sample: {}'.format(sound.sample_width * 8))


def audio_split(audio_file, output_path, split_length, target_format, verbose=False):
    # check audio file format
    audio_suffix = audio_file.split('.')[-1].lower()
    assert (audio_suffix in ['wav', 'mp3', 'ogg', 'flv', 'flac', 'aif', 'ape']), 'unsupported audio format: {}'.format(audio_file)

    sound = AudioSegment.from_file(audio_file)

    if verbose:
        print('Origin audio file info:')
        show_audio_info(sound)

    # split audio into segments
    chunks = make_chunks(sound, split_length)

    for i, chunk in enumerate(chunks):
        output_file = os.path.splitext(os.path.basename(audio_file))[0] + '_{}.'.format(i) + target_format
        chunk_name = os.path.join(output_path, output_file)

        chunk.export(chunk_name, format=target_format)

    if verbose:
        print('\nSplit chunk number:', len(chunks))


def main():
    parser = argparse.ArgumentParser(description='split audio files to target length segments')
    parser.add_argument('--audio_path', type=str, required=True,
                        help='audio file or directory to convert')
    parser.add_argument('--output_path', type=str, required=True,
                        help='output path to save target audio file')
    parser.add_argument('--split_length', type=int, required=False, default=1500,
                        help='target splited audio length in ms. default=%(default)s')
    parser.add_argument('--target_format', type=str, required=False, default='wav', choices=['wav', 'mp3', 'ogg'],
                        help='target audio file format. default=%(default)s')

    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    # get audio file list or single audio
    if os.path.isfile(args.audio_path):
        audio_split(args.audio_path, args.output_path, args.split_length, args.target_format, verbose=True)
    else:
        audio_files = glob.glob(os.path.join(args.audio_path, '*'))
        pbar = tqdm(total=len(audio_files), desc='Audio Split')

        for audio_file in audio_files:
            audio_split(audio_file, args.output_path, args.split_length, args.target_format, verbose=False)
            pbar.update(1)
        pbar.close()

    print('\nSplit finished')



if __name__ == "__main__":
    main()
