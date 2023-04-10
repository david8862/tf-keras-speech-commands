#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check & save voice segments in wav files with Conv-VAD model
"""
import os, sys, argparse
import glob
from scipy.io import wavfile
import numpy as np
from tqdm import tqdm

# install conv_vad with
# pip install https://github.com/sshh12/Conv-VAD/releases/download/v0.1.1/conv-vad-0.1.1.tar.gz
import conv_vad


def vad_clip(wav_file, vad, score_threshold, output_path):
    # load wav as numpy array
    sample_rate, audio = wavfile.read(wav_file)

    # check channel number, sample rate and sample depth
    # conv_vad only supports 1 channel audio with 16000 sample rate & 16 bit sample depth
    assert (len(audio.shape) == 1), 'conv_vad only supports single channle audio'
    assert (audio.dtype == np.int16), 'conv_vad only support 16 bit sample depth audio'
    assert (sample_rate == 16000), 'conv_vad only supports 16k sample rate audio'

    voice_detected = False
    voice_segment = np.array([], dtype=audio.dtype)

    for i in range(0, audio.shape[0]-sample_rate, sample_rate):
        audio_frame = audio[i : i+sample_rate]
        # For each audio frame (1 sec) compute the speech score.
        score = vad.score_speech(audio_frame)
        #print('Time =', i // sample_rate)
        #print('Speech Score: ', score)

        if score >= score_threshold:
            # found frame with voice
            voice_detected = True
            voice_segment = np.hstack((voice_segment, audio_frame))

        elif voice_detected == True:
            # end of a voice segment. save it and record end time
            time = i // sample_rate
            output_file = os.path.splitext(os.path.basename(wav_file))[0] + '_' + str(time) + '.wav'
            output_file = os.path.join(output_path, output_file)
            wavfile.write(output_file, sample_rate, voice_segment)

            # reset voice segment & flag
            voice_detected = False
            voice_segment = np.array([], dtype=audio.dtype)




def main():
    parser = argparse.ArgumentParser(description='check & save voice segments in wav files with Conv-VAD model')
    parser.add_argument('--wav_path', type=str, required=True,
                        help='wav file or directory to check')
    parser.add_argument('--score_threshold', type=float, required=False, default=0.7,
                        help='Conv-VAD score threshold for detecting voice. default=%(default)s')
    parser.add_argument('--output_path', type=str, required=True,
                        help='output path to save voice segment file')

    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    # create VAD object for audio process
    vad = conv_vad.VAD()

    # get wav audio file list or single audio
    if os.path.isfile(args.wav_path):
        vad_clip(args.wav_path, vad, args.score_threshold, args.output_path)
    else:
        wav_files = glob.glob(os.path.join(args.wav_path, '*.wav'))

        pbar = tqdm(total=len(wav_files), desc='VAD clip')
        for wav_file in wav_files:
            vad_clip(wav_file, vad, args.score_threshold, args.output_path)
            pbar.update(1)
        pbar.close()

    print('\nDone')


if __name__ == "__main__":
    main()
