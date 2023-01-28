#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyAudio Example: Record Audio to WAVE file
Reference from:
https://www.cnblogs.com/xiaosongshine/p/11088358.html
"""
import os, sys, argparse
import pyaudio
import wave
from tqdm import tqdm


def show_audio_info(wf):
    print('channels: {}'.format(wf.getnchannels()))
    print('sample rate: {}'.format(wf.getframerate()))
    print('bits per sample: {}'.format(wf.getsampwidth() * 8))
    print('total frames: {}'.format(wf.getnframes()))
    print('duration seconds: {} s'.format(wf.getnframes() / wf.getframerate()))
    #print('compress type: {}'.format(wf.getcomptype()))
    #print('compress name: {}'.format(wf.getcompname()))


def audio_record(output_file, channels, sample_rate, sample_bit, record_length, chunk_size):
    # create PyAudio stream
    p = pyaudio.PyAudio()
    pa_format = p.get_format_from_width(sample_bit//8)
    stream = p.open(format=pa_format,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=chunk_size)

    # create output wav file
    wf = wave.open(output_file, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(pa_format))
    wf.setframerate(sample_rate)

    # recording audio
    print('\nStart recording')
    for i in tqdm(range(0, int((sample_rate/chunk_size) * (record_length/1000)))):
        data = stream.read(chunk_size)
        wf.writeframes(data)

    stream.stop_stream()
    stream.close()
    p.terminate()
    print('Recording done.')

    # show recorded wav file info
    print('\nRecorded audio file info:')
    show_audio_info(wf)
    wf.close()



def main():
    parser = argparse.ArgumentParser(description='record audio from system microphone to wav file')
    parser.add_argument('--channels', type=int, required=False, default=1,
                        help='record audio channel number. default=%(default)s')
    parser.add_argument('--sample_rate', type=int, required=False, default=16000, choices=[8000, 16000, 22050, 44100, 48000],
                        help='record sample rate. default=%(default)s')
    parser.add_argument('--sample_bit', type=int, required=False, default=16, choices=[8, 16, 24, 32],
                        help='record sample bit number. default=%(default)s')
    parser.add_argument('--record_length', type=int, required=False, default=1500,
                        help='record audio length in ms. default=%(default)s')
    parser.add_argument('--chunk_size', type=int, required=False, default=1024,
                        help='record audio frame chunk size. default=%(default)s')
    parser.add_argument('--output_file', type=str, required=True,
                        help='output audio file')

    args = parser.parse_args()

    audio_record(args.output_file, args.channels, args.sample_rate, args.sample_bit, args.record_length, args.chunk_size)



if __name__ == "__main__":
    main()

