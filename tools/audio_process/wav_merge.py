#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge voice & noise into 1 audio file
"""
import os, sys, argparse
import numpy as np
from math import sqrt
from random import random
import wave, wavio


def load_audio(audio_file, sample_rate):
    wav = wavio.read(audio_file)

    if wav.rate != sample_rate:
        raise ValueError('Unsupported sample rate: ' + str(wav.rate))
    if wav.data.dtype != np.int16:
        raise ValueError('Unsupported data type: ' + str(wav.data.dtype))

    data = np.squeeze(wav.data)
    return data.astype(np.float32) / float(np.iinfo(data.dtype).max)


def save_audio(filename, audio, sample_rate):
    """
    Save loaded audio to file using the configured audio parameters
    """
    save_audio = (audio * np.iinfo(np.int16).max).astype(np.int16)
    wavio.write(filename, save_audio, sample_rate, sampwidth=2, scale='none')


def calc_volume(sample):
    """
    Find the RMS of the audio
    """
    return sqrt(np.mean(np.square(sample)))


def normalize_volume_to(sample, volume):
    """
    Normalize the volume to a certain RMS
    """
    return volume * sample / calc_volume(sample)


def chunk_audio(audio, chunk_size):
    chunk_list = []
    for i in range(chunk_size, len(audio), chunk_size):
        chunk_list.append(audio[i - chunk_size:i])

    return chunk_list


def merge(a, b, ratio):
    """
    Perform a weighted sum of a and b. ratio=1.0 means 100% of a and 0% of b
    """
    return ratio * a + (1.0 - ratio) * b


def wav_merge(voice_file, noise_file, voice_ratio, sample_rate, chunk_size, output_file):
    # load voice & noise audio
    voice_audio = load_audio(voice_file, sample_rate)
    noise_audio = load_audio(noise_file, sample_rate)

    # calculate adjusted voice volume
    voice_volume = calc_volume(voice_audio)
    #voice_volume *= 0.4 + 0.5 * random()

    # calculate adjusted noise volume
    noise_volume = calc_volume(noise_audio)
    #noise_volume *= 0.4 + 0.5 * random()

    # normalize voice & noise audio
    voice_audio = normalize_volume_to(voice_audio, noise_volume)
    noise_audio = normalize_volume_to(noise_audio, noise_volume)

    # crop audio to chunk list
    chunked_voice = chunk_audio(voice_audio, chunk_size)
    chunked_noise = chunk_audio(noise_audio, chunk_size)

    # merge voice & noise into 1 audio
    merged_buffer = np.array([], dtype=float)
    merged_length = len(chunked_voice) if len(chunked_voice) < len(chunked_noise) else len(chunked_noise)
    for i in range(merged_length):
        chunk = merge(chunked_voice[i], chunked_noise[i], voice_ratio)
        merged_buffer = np.concatenate((merged_buffer, chunk))

    # save merged audio
    save_audio(output_file, merged_buffer, sample_rate)



def main():
    parser = argparse.ArgumentParser(description='merge .wav voice audio & background noise audio file. only support 16 bit format')
    parser.add_argument('--voice_file', type=str, required=True,
                        help='voice audio file')
    parser.add_argument('--noise_file', type=str, required=True,
                        help='background noise audio file')
    parser.add_argument('--voice_ratio', type=float, required=False, default=0.6,
                        help='voice ratio in merged audio. default=%(default)s')
    parser.add_argument('--sample_rate', type=int, required=False, default=16000, choices=[8000, 16000, 22050, 44100, 48000],
                        help='audio sample rate. default=%(default)s')
    parser.add_argument('--chunk_size', type=int, required=False, default=1024,
                        help='audio frame chunk size. default=%(default)s')
    parser.add_argument('--output_file', type=str, required=True,
                        help='merged audio file')

    args = parser.parse_args()

    wav_merge(args.voice_file, args.noise_file, args.voice_ratio, args.sample_rate, args.chunk_size, args.output_file)



if __name__ == "__main__":
    main()
