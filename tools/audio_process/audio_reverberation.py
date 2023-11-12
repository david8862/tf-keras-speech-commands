#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A simple demo for simulating acoustics reverberation with "Pyroomacoustics"

Reference from:
https://www.cnblogs.com/LXP-Never/p/13404523.html
https://zhuanlan.zhihu.com/p/340603035
https://zhuanlan.zhihu.com/p/524445325
https://www.cnblogs.com/tingweichen/p/13861569.html


RIR simulation tools:
https://cloud.tencent.com/developer/news/981007

rir_generator
https://github.com/audiolabs/rir-generator (python)
https://github.com/ehabets/RIR-Generator (matlab)

pyroomacoustics
https://github.com/LCAV/pyroomacoustics
https://pyroomacoustics.readthedocs.io/en/pypi-release/pyroomacoustics.room.html
https://notebook.community/LCAV/pyroomacoustics/notebooks/pyroomacoustics_demo

gpuRIR
https://github.com/DavidDiazGuerra/gpuRIR
https://mp.weixin.qq.com/s/q8iBh2OO-Qz1wT7J1Uix7A
"""
import os, sys, argparse
import glob
from tqdm import tqdm
from random import random, choice
from shutil import copy
import pyroomacoustics as pra
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import librosa


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


def audio_reverberation(voice_file, noise_file, sample_rate, output_path, noised_rate=1.0, visualize=False):
    # 1. 创建房间
    # 所需的混响时间和房间的尺寸
    RT60 = Parameter(0.3, 0.7).getvalue() # 所需的混响时间, 秒

    # 我们定义了一个6m x 4.8m x 2.8m的房间(28.8m2, 中等规模家居客厅场景)
    room_sz = Parameter([4, 3, 2.6], [6, 4.8, 2.8]).getvalue() # 此时随机得到[4, 3, 2.6]~[6, 4.8, 2.8]之间的一个房间尺寸
    room_length = room_sz[0]
    room_width = room_sz[1]
    #room_corner = np.array([[0, 0], [room_length, 0], [room_length, room_width], [0, room_width]]).T
    #room_height = room_sz[2]

    # 我们可以使用Sabine’s公式来计算壁面能量吸收和达到预期混响时间所需的ISM的最大阶数(RT60, 即RIR衰减60分贝所需的时间)
    e_absorption, max_order = pra.inverse_sabine(RT60, room_sz)    # 返回 墙壁吸收的能量 和 允许的反射次数
    # 我们还可以自定义 墙壁材料 和 最大反射次数
    # m = pra.Material(energy_absorption="hard_surface")    # 定义 墙的材料, 我们还可以定义不同墙面的的材料
    # max_order = 3

    room = pra.ShoeBox(room_sz, absorption=None, fs=sample_rate, materials=pra.Material(e_absorption), max_order=max_order, ray_tracing=False, air_absorption=False)
    #room = pra.Room.from_corners(room_corner, absorption=None, fs=sample_rate, materials=pra.Material(e_absorption), max_order=max_order)
    #room.extrude(height=room_height, v_vec=None, absorption=None, materials=pra.Material(e_absorption))

    # 激活射线追踪
    #room.set_ray_tracing()


    # 2. 创建语音声源
    # 在房间内创建一个位于[3.0, 2.4, 1.70]的语音声源(房间正中，一人高), 从0.3秒开始向仿真中发出wav文件的内容
    voice_sr, voice_data = wavfile.read(voice_file)
    #voice_data, voice_sr = librosa.load(voice_file, sr=sample_rate)  # 导入一个单通道音频作为语音声源信号
    assert voice_data.ndim == 1, 'only support single channel audio for voice file'
    assert voice_sr == sample_rate, 'sample rate mismatch for voice audio {}'.format(voice_file)

    voice_pos = Parameter([0.5, 0.5, 1.6], [room_length-0.5, room_width-0.5, 1.9]).getvalue() # 此时随机得到[0.5, 0.5, 1.6]~[room_length-0.5, room_width-0.5, 1.9]之间的一个声源坐标
    room.add_source(voice_pos, signal=voice_data, delay=0)


    # 3. 在房间放置麦克风
    # 定义麦克风的位置: (ndim, nmics) 即每个列包含一个麦克风的坐标
    # 在这里我们创建一个带有三个麦克风(圆形三麦阵列, 直径4cm, 中心坐标[1.5, 1.2, 0.1])的数组
    # 三个麦克风分别位于
    # [1.5,    1.18, 0.1]
    # [1.4827, 1.21, 0.1]
    # [1.5173, 1.21, 0.1]
    mic_height = 0.1
    mic_num = 3
    mic_radius = 0.02

    # create 2D circle pos array for mics
    mic_center = Parameter([0.5, 0.5], [room_length-0.5, room_width-0.5]).getvalue()
    R = pra.circular_2D_array(center=mic_center, M=mic_num, phi0=0, radius=mic_radius)
    # add height in z-axis
    mic_pos = np.r_[R, np.array([[mic_height]*mic_num])]


    #mic_center = Parameter([0.5, 0.5, mic_height], [room_length-0.5, room_width-0.5, mic_height]).getvalue()
    #mic_bias = np.array([
        #[0,      -0.02, 0],  # mic1
        #[-0.0173, 0.01, 0],  # mic2
        #[0.0173,  0.01, 0],  # mic3
    #])
    #mic_pos = (mic_center + mic_bias).T

    room.add_microphone_array(mic_pos, directivity=None)     # 最后将麦克风阵列放在房间里


    # 4. 创建噪声声源
    if noise_file and random() < noised_rate:
        # 在房间内创建一个位于[1.5, 1.38, 0.1]的噪声声源(与麦克风阵列距离很近), 从0.3秒开始向仿真中发出wav文件的内容
        noise_sr, noise_data = wavfile.read(noise_file)
        #noise_data, noise_sr = librosa.load(noise_file, sr=sample_rate)  # 导入一个单通道音频作为噪声声源信号
        assert noise_data.ndim == 1, 'only support single channel audio for noise file'
        assert noise_sr == sample_rate, 'sample rate mismatch for noise audio {}'.format(noise_file)


        noise_bias = np.array([0, 0.18, 0])
        noise_pos = mic_center + noise_bias
        room.add_source(noise_pos, signal=noise_data, delay=0)


    # 5. 创建房间冲击响应(Room Impulse Response)
    room.compute_rir()
    #room.image_source_model()

    # 6. 模拟声音传播, 每个源的信号将与相应的房间脉冲响应进行卷积, 卷积的输出将在麦克风上求和
    room.simulate()
    #room.simulate(reference_mic=0, snr=10)      # 控制信噪比

    # 保存所有的信号到wav文件
    output_file = os.path.join(output_path, os.path.splitext(os.path.basename(voice_file))[0]+'_reverb.wav')
    room.mic_array.to_wav(output_file, norm=False, bitdepth=np.int16)

    if visualize:
        # 测量混响时间
        rt60 = room.measure_rt60()
        print("The desired RT60 was {}".format(RT60))
        print("The measured RT60 is {}".format(rt60[1, 0]))


        plt.figure()
        # 绘制其中一个RIR. both can also be plotted using room.plot_rir()
        rir_1_0 = room.rir[1][0]    # 画出mic1和source0之间的RIR
        plt.subplot(2, 1, 1)
        plt.plot(np.arange(len(rir_1_0)) / room.fs, rir_1_0)
        plt.title("The RIR from source 0 to mic 1")
        plt.xlabel("Time [s]")

        # 绘制mic1 处接收到的信号
        plt.subplot(2, 1, 2)
        plt.plot(np.arange(len(room.mic_array.signals[1, :])) / room.fs, room.mic_array.signals[1, :])
        plt.title("Microphone 1 signal")
        plt.xlabel("Time [s]")

        plt.tight_layout()
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='simulate acoustics reverberation with Pyroomacoustics')
    parser.add_argument('--voice_path', type=str, required=True,
                        help='voice audio file or directory for simulate')
    parser.add_argument('--noise_path', type=str, required=False, default=None,
                        help='noice audio file or directory for simulate, default=%(default)s')
    parser.add_argument('--noised_rate', type=float, required=False, default=1.0,
                        help='random percentage rate of adding noise in simulation (0.0~1.0). default=%(default)s')
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

        audio_reverberation(args.voice_path, noise_file, args.sample_rate, args.output_path)

    else:
        voice_files = glob.glob(os.path.join(args.voice_path, '*.wav'))
        pbar = tqdm(total=len(voice_files), desc='Process voice files')

        for voice_file in voice_files:
            # random pick noise file
            noise_file = choice(noise_files)

            audio_reverberation(voice_file, noise_file, args.sample_rate, args.output_path, args.noised_rate)

            pbar.update(1)
        pbar.close()
    print('Done.')


if __name__ == "__main__":
    main()
