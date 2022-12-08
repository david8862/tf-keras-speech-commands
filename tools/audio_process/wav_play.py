#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyAudio Example: Play WAVE file
Reference from:
https://blog.csdn.net/dss_dssssd/article/details/83540061
"""
import os, sys, argparse
import glob
import pyaudio
import wave
from tqdm import tqdm


def show_audio_info(wf):
    print('channels: {}'.format(wf.getnchannels()))
    print('sample rate: {}'.format(wf.getframerate()))
    print('bits per sample: {}'.format(wf.getsampwidth() * 8))
    print('total frames: {}'.format(wf.getnframes()))
    print('duration seconds: {} s'.format(wf.getnframes() / wf.getframerate()))
    print('compress type: {}'.format(wf.getcomptype()))
    print('compress name: {}'.format(wf.getcompname()))


def wav_play(wav_file, chunk_size):
    wf = wave.open(wav_file, 'rb')

    print('\nAudio file info:')
    show_audio_info(wf)

    # instantiate PyAudio (1)
    p = pyaudio.PyAudio()

    # open stream (2)
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    # read data
    data = wf.readframes(chunk_size)

    # play stream (3)
    datas = []
    while len(data) > 0:
        data = wf.readframes(chunk_size)
        datas.append(data)

    print('\nStart playing')
    for d in tqdm(datas):
        stream.write(d)

    # stop stream (4)
    stream.stop_stream()
    stream.close()

    # close PyAudio (5)
    wf.close()
    p.terminate()
    print('Playing done.')



def main():
    parser = argparse.ArgumentParser(description='play wav audio files')
    parser.add_argument('--wav_path', type=str, required=True,
                        help='wav file or directory to play')
    parser.add_argument('--chunk_size', type=int, required=False, default=1024,
                        help='audio frame chunk size. default=%(default)s')

    args = parser.parse_args()

    # get wav audio file list or single audio
    if os.path.isfile(args.wav_path):
        wav_play(args.wav_path, args.chunk_size)
    else:
        wav_files = glob.glob(os.path.join(args.wav_path, '*'))

        for i, wav_file in enumerate(wav_files):
            print('\nPlaying ({}/{}): {}'.format(i+1, len(wav_files), wav_file))
            wav_play(wav_file, args.chunk_size)

    print('\nDone')



if __name__ == "__main__":
    main()

