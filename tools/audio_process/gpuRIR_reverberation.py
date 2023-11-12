#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A simple application for simulating acoustics reverberation with "gpuRIR"

Reference from:
https://github.com/DavidDiazGuerra/gpuRIR/blob/master/examples/example.py
https://mp.weixin.qq.com/s/q8iBh2OO-Qz1wT7J1Uix7A

install gpuRIR with following cmd (need CUDA support):
$ pip install https://github.com/DavidDiazGuerra/gpuRIR/zipball/master
"""
import os, sys, argparse
import glob
from tqdm import tqdm
from random import random, choice
from shutil import copy
import numpy as np
import soundfile
#from scipy.io import wavfile
import gpuRIR

gpuRIR.activateMixedPrecision(False)
gpuRIR.activateLUT(False)



# 生成(min, max)之间的一个随机值
class Parameter:
    def __init__(self, *args):
        if len(args) == 1:
            self.random = False
            self.value = np.array(args[0])
            self.min_value = None
            self.max_value = None
        elif len(args) == 2:
            self.random = True
            self.min_value = np.array(args[0])
            self.max_value = np.array(args[1])
            self.value = None
        else:
            raise Exception(
                'Parammeter must be called with one (value) or two (min and max value) array_like parammeters')
    def getvalue(self):
        if self.random:
            return self.min_value + np.random.random(self.min_value.shape) * (self.max_value - self.min_value)
        else:
            return self.value


def gpuRIR_reverberation(voice_file, noise_file, sample_rate, output_path):
    # 1. 创建房间
    # 所需的混响时间和房间的尺寸
    RT60 = Parameter(0.3, 0.7).getvalue() # 所需的混响时间, 秒

    # 我们定义了一个6m x 4.8m x 2.8m的房间(28.8m2, 中等规模家居客厅场景)
    room_sz = Parameter([4, 3, 2.6], [6, 4.8, 2.8]).getvalue() # 此时随机得到[4, 3, 2.6]~[6, 4.8, 2.8]之间的一个房间尺寸
    room_length = room_sz[0]
    room_width = room_sz[1]

    att_diff = 15.0 # Attenuation when start using the diffuse reverberation model [dB]
    att_max = 60.0 # Attenuation at the end of the simulation [dB]

    beta = gpuRIR.beta_SabineEstimation(room_sz, RT60) # Reflection coefficients
    Tdiff= gpuRIR.att2t_SabineEstimator(att_diff, RT60) # Time to start the diffuse reverberation model [s]
    Tmax = gpuRIR.att2t_SabineEstimator(att_max, RT60) # Time to stop the simulation [s]
    nb_img = gpuRIR.t2n(Tdiff, room_sz) # Number of image sources in each dimension


    # 2. 创建语音声源
    # 在房间内创建一个位于[3.0, 2.4, 1.70]的语音声源(房间正中，一人高), 从0.3秒开始向仿真中发出wav文件的内容
    voice_pos = Parameter([0.5, 0.5, 1.6], [room_length-0.5, room_width-0.5, 1.9]).getvalue() # 此时随机得到[0.5, 0.5, 1.6]~[room_length-0.5, room_width-0.5, 1.9]之间的一个声源坐标
    source_pos = [voice_pos]

    # read voice data
    voice_data, voice_sr = soundfile.read(voice_file)
    #voice_sr, voice_data = wavfile.read(voice_file)
    assert voice_data.ndim == 1, 'only support single channel audio for voice file'
    assert voice_sr == sample_rate, 'sample rate mismatch for voice audio {}'.format(voice_file)
    data = voice_data


    # 3. 在房间放置麦克风
    # 定义麦克风的位置: (ndim, nmics) 即每个列包含一个麦克风的坐标
    # 在这里我们创建一个带有三个麦克风(圆形三麦阵列, 直径4cm, 中心坐标[1.5, 1.2, 0.1])的数组
    # 三个麦克风分别位于
    # [1.5,    1.18, 0.1]
    # [1.4827, 1.21, 0.1]
    # [1.5173, 1.21, 0.1]
    mic_height = 0.1
    mic_center = Parameter([0.5, 0.5, mic_height], [room_length-0.5, room_width-0.5, mic_height]).getvalue()

    mic_num = 3
    mic_bias = np.array([
        [0,      -0.02, 0],  # mic1
        [-0.0173, 0.01, 0],  # mic2
        [0.0173,  0.01, 0],  # mic3
    ])
    mic_pos = (mic_center + mic_bias)


    # 4. 创建噪声声源
    if noise_file and random() < noised_rate:
        # 在房间内创建一个位于[1.5, 1.38, 0.1]的噪声声源(与麦克风阵列距离很近), 从0.3秒开始向仿真中发出wav文件的内容
        noise_bias = np.array([0, 0.18, 0])
        noise_pos = mic_center + noise_bias
        source_pos.append(noise_pos)

        # read noise data
        noise_data, noise_sr = soundfile.read(noise_file)
        #noise_sr, noise_data = wavfile.read(noise_file)
        assert noise_data.ndim == 1, 'only support single channel audio for noise file'
        assert noise_sr == sample_rate, 'sample rate mismatch for noise audio {}'.format(noise_file)

        # align noise length with voice, for simulateTrajectory
        if len(noise_data) > len(voice_data):
            noise_data = noise_data[:len(voice_data)]
        elif len(noise_data) < len(voice_data):
            padding_len = len(voice_data) - len(noise_data)
            noise_data = np.pad(noise_data, ((0, padding_len)), 'constant', constant_values=(0,0))
        data = np.array(voice_data + noise_data)


    # 5. 生成RIR
    RIR = gpuRIR.simulateRIR(
        room_sz=room_sz,
        beta=beta,
        pos_src=np.array(source_pos),
        pos_rcv=mic_pos,
        nb_img=nb_img,
        Tmax=Tmax,
        fs=sample_rate,
        Tdiff=Tdiff,
        mic_pattern='omni'
    )

    # 6. 生成多通道语音
    reverb_data = gpuRIR.simulateTrajectory(data, RIR, fs=sample_rate)

    # save simulated audio
    output_file = os.path.join(output_path, os.path.splitext(os.path.basename(voice_file))[0]+'_reverb.wav')
    soundfile.write(output_file, reverb_data, sample_rate)
    #wavfile.write(output_file, sample_rate, reverb_data)


def main():
    parser = argparse.ArgumentParser(description='simulate acoustics reverberation with gpuRIR')
    parser.add_argument('--voice_path', type=str, required=True,
                        help='voice audio file or directory for simulate')
    parser.add_argument('--noise_path', type=str, required=False, default=None,
                        help='noice audio file or directory for simulate, default=%(default)s')
    parser.add_argument('--sample_rate', type=int, required=False, default=16000, choices=[8000, 16000, 22050, 44100, 48000],
                        help='audio sample rate. default=%(default)s')
    parser.add_argument('--output_path', type=str, required=True,
                        help='output path to save simulated audio file, default=%(default)s')

    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    # get noise audio file list or single noise audio
    if args.noise_path is not None and os.path.isdir(args.noise_path):
        noise_files = glob.glob(os.path.join(args.noise_path, '*.wav'))
    else:
        noise_files = [args.noise_path]

    # process single voice file, or loop for voice file list
    if os.path.isfile(args.voice_path):
        # random pick noise file
        noise_file = choice(noise_files)

        gpuRIR_reverberation(args.voice_path, noise_file, args.sample_rate, args.output_path)

    else:
        voice_files = glob.glob(os.path.join(args.voice_path, '*.wav'))
        pbar = tqdm(total=len(voice_files), desc='Process voice files')

        for voice_file in voice_files:
            # random pick noise file
            noise_file = choice(noise_files)

            gpuRIR_reverberation(voice_file, noise_file, args.sample_rate, args.output_path)

            pbar.update(1)
        pbar.close()
    print('Done.')


if __name__ == "__main__":
    main()
