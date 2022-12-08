#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reference from:
https://blog.csdn.net/pengranxindong/article/details/90606279
"""
import os, sys, argparse
import glob
from pydub import AudioSegment
from pydub.playback import play


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


def audio_play(audio_file):
    # check audio file format
    audio_suffix = audio_file.split('.')[-1].lower()
    assert (audio_suffix in ['wav', 'mp3', 'ogg', 'flv', 'flac', 'aif', 'ape']), 'unsupported audio format: {}'.format(audio_file)

    sound = AudioSegment.from_file(audio_file)

    print('\nAudio file info:')
    show_audio_info(sound)

    play(sound)
    print('Playing done.')



def main():
    parser = argparse.ArgumentParser(description='play audio files')
    parser.add_argument('--audio_path', type=str, required=True,
                        help='audio file or directory to play')

    args = parser.parse_args()

    # get audio file list or single audio
    if os.path.isfile(args.audio_path):
        audio_play(args.audio_path)
    else:
        audio_files = glob.glob(os.path.join(args.audio_path, '*'))

        for i, audio_file in enumerate(audio_files):
            print('\nPlaying ({}/{}): {}'.format(i+1, len(audio_files), audio_file))
            audio_play(audio_file)

    print('\nDone')



if __name__ == "__main__":
    main()

