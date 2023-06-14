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
https://pyroomacoustics.readthedocs.io/en/pypi-release/

gpuRIR
https://github.com/DavidDiazGuerra/gpuRIR
"""
import os, sys, argparse
import pyroomacoustics as pra
import numpy as np
import matplotlib.pyplot as plt
import librosa


def audio_reverberation(voice_file, noise_file, sample_rate, output_file):
    # 1. 创建房间

    # 所需的混响时间和房间的尺寸
    rt60_tgt = 0.5  # 所需的混响时间, 秒

    # 我们定义了一个6m x 4.8m x 2.8m的房间(28.8m2, 中等规模家居客厅场景)
    room_dim = [6, 4.8, 2.8]
    room_corner = np.array([[0, 0], [6, 0], [6, 4.8], [0, 4.8]]).T
    room_height = 2.8

    # 我们可以使用Sabine’s公式来计算壁面能量吸收和达到预期混响时间所需的ISM的最大阶数(RT60, 即RIR衰减60分贝所需的时间)
    e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)    # 返回 墙壁吸收的能量 和 允许的反射次数
    # 我们还可以自定义 墙壁材料 和 最大反射次数
    # m = pra.Material(energy_absorption="hard_surface")    # 定义 墙的材料, 我们还可以定义不同墙面的的材料
    # max_order = 3

    room = pra.ShoeBox(room_dim, absorption=None, fs=sample_rate, materials=pra.Material(e_absorption), max_order=max_order, ray_tracing=False, air_absorption=False)
    #room = pra.Room.from_corners(room_corner, absorption=None, fs=sample_rate, materials=pra.Material(e_absorption), max_order=max_order)
    #room.extrude(height=room_height, v_vec=None, absorption=None, materials=pra.Material(e_absorption))


    # 激活射线追踪
    #room.set_ray_tracing()

    # 2. 创建声源
    # 在房间内创建一个位于[3.0, 2.4, 1.70]的语音声源(房间正中，一人高), 从0.3秒开始向仿真中发出wav文件的内容
    voice_data, _ = librosa.load(voice_file, sr=sample_rate)  # 导入一个单通道音频作为语音声源信号
    room.add_source([3.0, 2.4, 1.70], signal=voice_data, delay=0.3)

    if noise_file:
        # 在房间内创建一个位于[1.5, 1.38, 0.1]的噪声声源(与麦克风阵列距离很近), 从0.3秒开始向仿真中发出wav文件的内容
        noise_data, _ = librosa.load(noise_file, sr=sample_rate)  # 导入一个单通道音频作为噪声声源信号
        room.add_source([1.5, 1.38, 0.1], signal=noise_data, delay=0.3)

    # 3. 在房间放置麦克风
    # 定义麦克风的位置: (ndim, nmics) 即每个列包含一个麦克风的坐标
    # 在这里我们创建一个带有三个麦克风(圆形三麦阵列, 直径4cm, 中心坐标[1.5, 1.2, 0.1])的数组
    # 三个麦克风分别位于
    # [1.5,    1.18, 0.1]
    # [1.4827, 1.21, 0.1]
    # [1.5173, 1.21, 0.1]
    mic_locs = np.c_[
        [1.5,    1.18, 0.1],  # mic1
        [1.4827, 1.21, 0.1],  # mic2
        [1.5173, 1.21, 0.1],  # mic3
    ]

    room.add_microphone_array(mic_locs, directivity=None)     # 最后将麦克风阵列放在房间里

    # 4. 创建房间冲击响应(Room Impulse Response)
    room.compute_rir()
    #room.image_source_model()

    # 5. 模拟声音传播, 每个源的信号将与相应的房间脉冲响应进行卷积, 卷积的输出将在麦克风上求和
    room.simulate()
    #room.simulate(reference_mic=0, snr=10)      # 控制信噪比

    # 保存所有的信号到wav文件
    room.mic_array.to_wav(output_file, norm=True, bitdepth=np.int16)

    # 测量混响时间
    rt60 = room.measure_rt60()
    print("The desired RT60 was {}".format(rt60_tgt))
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--voice_file', type=str, required=True,
                        help='voice audio file')
    parser.add_argument('--noise_file', type=str, required=False, default=None,
                        help='noise audio file')
    parser.add_argument('--sample_rate', type=int, required=False, default=16000, choices=[8000, 16000, 22050, 44100, 48000],
                        help='audio sample rate. default=%(default)s')
    parser.add_argument('--output_file', type=str, required=True,
                        help='merged audio file')

    args = parser.parse_args()


    audio_reverberation(args.voice_file, args.noise_file, args.sample_rate, args.output_file)



if __name__ == "__main__":
    main()
