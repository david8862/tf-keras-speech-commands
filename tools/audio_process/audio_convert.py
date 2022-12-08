#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reference from:
https://blog.csdn.net/zkw_1998/article/details/118360485
"""
import os, sys, argparse
import glob
from tqdm import tqdm
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
import wave
import scipy


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


def show_wav_info(wf):
    print('channels: {}'.format(wf.getnchannels()))
    print('sample rate: {}'.format(wf.getframerate()))
    print('bits per sample: {}'.format(wf.getsampwidth() * 8))
    print('total frames: {}'.format(wf.getnframes()))
    print('duration seconds: {} s'.format(wf.getnframes() / wf.getframerate()))
    #print('compress type: {}'.format(wf.getcomptype()))
    #print('compress name: {}'.format(wf.getcompname()))


def generate_white_noise(length, sample_rate, sample_bit=16, amplitude=0.7):
    amplitude_bit = int(sample_bit * amplitude)
    assert (sample_bit == 16), 'only support 16 bit sample for white noise generate'

    noise_data = scipy.stats.truncnorm(-1, 1, scale=min(2**sample_bit, 2**amplitude_bit)).rvs(int(sample_rate * (length / 1000.0)))
    noise_data = noise_data.astype(np.int16)
    return noise_data


def pcm_convert(audio_file, output_path, channel_num, sample_rate, sample_bit, clip_length, fill_white_noise, noise_amplitude, target_format, verbose=False):
    assert (channel_num and sample_rate and sample_bit), \
            'convert pcm audio {} need to provide channel number, sample_rate and sample_bit'.format(audio_file)

    assert (target_format == 'wav'), 'only support convert to .wav for pcm audio'
    assert (channel_num == 1), 'only support single channel for pcm audio'

    f = open(audio_file, 'rb')
    pcm_data = f.read()

    if clip_length:
        # pcm_data is a bytes array
        bytes_length = int(sample_rate * (clip_length / 1000.0) * (sample_bit//8))

        if bytes_length <= len(pcm_data):
            pcm_data = pcm_data[-bytes_length:]  # clip from tail
        else:
            if fill_white_noise:
                noise_length = clip_length - (len(pcm_data) * 1000 / (sample_rate * (sample_bit//8))) # calculate noise length in ms
                noise_data = generate_white_noise(noise_length, sample_rate, sample_bit, noise_amplitude)
                silent_data = noise_data.tobytes()
            else:
                silent_length = bytes_length - len(pcm_data)
                silent_data = bytes([0] * silent_length)

            pcm_data = silent_data + pcm_data  # add silent segment at head

    output_file = os.path.splitext(os.path.basename(audio_file))[0] + '.' + target_format
    output_file = os.path.join(output_path, output_file)
    wf = wave.open(output_file, 'wb')
    wf.setnchannels(channel_num)
    wf.setsampwidth(sample_bit//8)
    wf.setframerate(sample_rate)
    wf.writeframes(pcm_data)

    if verbose:
        print('\nConverted audio file info:')
        show_wav_info(wf)

    wf.close()


def audio_convert(audio_file, output_path, channel_num, sample_rate, sample_bit, clip_length, fill_white_noise, noise_amplitude, target_format, verbose=False):
    audio_suffix = audio_file.split('.')[-1].lower()

    # convert pcm raw audio data
    if audio_suffix in ['pcm', 'raw']:
        pcm_convert(audio_file, output_path, channel_num, sample_rate, sample_bit, clip_length, fill_white_noise, noise_amplitude, target_format, verbose)
        return

    # check audio file format
    assert (audio_suffix in ['wav', 'mp3', 'ogg', 'flv', 'flac', 'aif', 'ape']), 'unsupported audio format: {}'.format(audio_file)

    sound = AudioSegment.from_file(audio_file)

    if verbose:
        print('Origin audio file info:')
        show_audio_info(sound)

    if channel_num:
        sound = sound.set_channels(channel_num)
    else:
        channel_num = sound.channels

    if sample_rate:
        sound = sound.set_frame_rate(sample_rate)
    else:
        sample_rate = sound.frame_rate

    if sample_bit:
        sample_width = sample_bit // 8
        sound = sound.set_sample_width(sample_width)
    else:
        sample_bit = sound.sample_width * 8

    if clip_length:
        # check if clip length longer than total audio length
        if clip_length <= sound.duration_seconds * 1000:
            sound = sound[-clip_length:]  # clip from tail
        else:
            silent_length = clip_length - sound.duration_seconds * 1000
            if fill_white_noise:
                noise_data = generate_white_noise(silent_length, sample_rate, sample_bit, noise_amplitude)
                sound_silent = AudioSegment(noise_data.tobytes(),
                                            frame_rate=sample_rate,
                                            sample_width=noise_data.dtype.itemsize,
                                            channels=1)
            else:
                sound_silent = AudioSegment.silent(duration=silent_length)

            sound = sound_silent + sound  # add silent segment at head

    if verbose:
        print('\nConverted audio file info:')
        show_audio_info(sound)

    output_file = os.path.splitext(os.path.basename(audio_file))[0] + '.' + target_format
    output_file = os.path.join(output_path, output_file)
    sound.export(output_file, format=target_format)



def main():
    parser = argparse.ArgumentParser(description='convert audio files to target format')
    parser.add_argument('--audio_path', type=str, required=True,
                        help='audio file or directory to convert')
    parser.add_argument('--output_path', type=str, required=True,
                        help='output path to save target audio file')
    parser.add_argument('--channel_num', type=int, required=False, default=None,
                        help='target channel number, None is unchange. default=%(default)s')
    parser.add_argument('--sample_rate', type=int, required=False, default=None, choices=[None, 8000, 16000, 22050, 44100, 48000],
                        help='target sample rate, None is unchange. default=%(default)s')
    parser.add_argument('--sample_bit', type=int, required=False, default=None, choices=[None, 8, 16, 24, 32],
                        help='target sample bit number, None is unchange. default=%(default)s')
    parser.add_argument('--clip_length', type=int, required=False, default=None,
                        help='target audio length in ms, None is unchange. default=%(default)s')
    parser.add_argument('--fill_white_noise', default=False, action="store_true",
                        help='fill appended silent audio segment with white noise')
    parser.add_argument('--noise_amplitude', type=float, required=False, default=0.7,
                        help='white noise amplitude. default=%(default)s')
    parser.add_argument('--target_format', type=str, required=False, default='wav', choices=['wav', 'mp3'],
                        help='target audio file format. default=%(default)s')

    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    # get audio file list or single audio
    if os.path.isfile(args.audio_path):
        audio_convert(args.audio_path, args.output_path, args.channel_num, args.sample_rate, args.sample_bit, args.clip_length, args.fill_white_noise, args.noise_amplitude, args.target_format, verbose=True)
    else:
        audio_files = glob.glob(os.path.join(args.audio_path, '*'))
        pbar = tqdm(total=len(audio_files), desc='Audio Convert')

        for audio_file in audio_files:
            audio_convert(audio_file, args.output_path, args.channel_num, args.sample_rate, args.sample_bit, args.clip_length, args.fill_white_noise, args.noise_amplitude, args.target_format, verbose=False)
            pbar.update(1)
        pbar.close()

    print('\nConvert finished')



if __name__ == "__main__":
    main()
