#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Split wakeword section from roborock raw speech command wav audios, using VAD algorithm
"""
import os, sys, argparse
import glob
import numpy as np
import scipy.io.wavfile as wf
import matplotlib.pyplot as plt
from shutil import copy
from tqdm import tqdm


class VoiceActivityDetector(object):
    """
    Use signal energy to detect voice activity in wav file
    """
    def __init__(self, wav_file):
        self._read_wav(wav_file)._convert_to_mono()
        self.sample_window = 0.02 #20 ms
        self.sample_overlap = 0.01 #10ms
        self.speech_window = 0.5 #half a second
        self.speech_energy_threshold = 0.6 #60% of energy in voice band
        self.speech_start_band = 300
        self.speech_end_band = 3000

    def _read_wav(self, wav_file):
        self.rate, self.data = wf.read(wav_file)
        self.channels = len(self.data.shape)
        self.filename = wav_file
        return self

    def _convert_to_mono(self):
        if self.channels == 2 :
            self.data = np.mean(self.data, axis=1, dtype=self.data.dtype)
            self.channels = 1
        return self

    def _calculate_frequencies(self, audio_data):
        data_freq = np.fft.fftfreq(len(audio_data),1.0/self.rate)
        data_freq = data_freq[1:]
        return data_freq

    def _calculate_amplitude(self, audio_data):
        data_ampl = np.abs(np.fft.fft(audio_data))
        data_ampl = data_ampl[1:]
        return data_ampl

    def _calculate_energy(self, data):
        data_amplitude = self._calculate_amplitude(data)
        data_energy = data_amplitude ** 2
        return data_energy

    def _znormalize_energy(self, data_energy):
        energy_mean = np.mean(data_energy)
        energy_std = np.std(data_energy)
        energy_znorm = (data_energy - energy_mean) / energy_std
        return energy_znorm

    def _connect_energy_with_frequencies(self, data_freq, data_energy):
        energy_freq = {}
        for (i, freq) in enumerate(data_freq):
            if abs(freq) not in energy_freq:
                energy_freq[abs(freq)] = data_energy[i] * 2
        return energy_freq

    def _calculate_normalized_energy(self, data):
        data_freq = self._calculate_frequencies(data)
        data_energy = self._calculate_energy(data)
        #data_energy = self._znormalize_energy(data_energy) #znorm brings worse results
        energy_freq = self._connect_energy_with_frequencies(data_freq, data_energy)
        return energy_freq

    def _sum_energy_in_band(self,energy_frequencies, start_band, end_band):
        sum_energy = 0
        for f in energy_frequencies.keys():
            if start_band<f<end_band:
                sum_energy += energy_frequencies[f]
        return sum_energy

    def _median_filter (self, x, k):
        assert k % 2 == 1, "Median filter length must be odd."
        assert x.ndim == 1, "Input must be one-dimensional."
        k2 = (k - 1) // 2
        y = np.zeros ((len (x), k), dtype=x.dtype)
        y[:,k2] = x
        for i in range (k2):
            j = k2 - i
            y[j:,i] = x[:-j]
            y[:j,i] = x[0]
            y[:-j,-(i+1)] = x[j:]
            y[-j:,-(i+1)] = x[-1]
        return np.median (y, axis=1)

    def _smooth_speech_detection(self, detected_windows):
        median_window=int(self.speech_window/self.sample_window)
        if median_window%2==0: median_window=median_window-1
        median_energy = self._median_filter(detected_windows[:,1], median_window)
        return median_energy

    def convert_windows_to_readable_labels(self, detected_windows):
        """
        Takes as input array of window numbers and speech flags from speech
        detection and convert speech flags to time intervals of speech.
        Output is array of dictionaries with speech intervals.
        """
        speech_time = []
        is_speech = 0
        for window in detected_windows:
            if (window[1]==1.0 and is_speech==0):
                is_speech = 1
                speech_label = {}
                speech_start_time = window[0] / self.rate
                speech_label['speech_begin'] = speech_start_time
                #print(window[0], speech_start_time)
                #speech_time.append(speech_label)
            if (window[1]==0.0 and is_speech==1):
                is_speech = 0
                speech_end_time = window[0] / self.rate
                speech_label['speech_end'] = speech_end_time
                speech_time.append(speech_label)
                #print(window[0], speech_end_time)
        return speech_time

    def plot_detected_speech_regions(self):
        """
        Performs speech detection and plot original signal and speech regions.
        """
        data = self.data
        detected_windows = self.detect_speech()
        data_speech = np.zeros(len(data))
        it = np.nditer(detected_windows[:,0], flags=['f_index'])
        while not it.finished:
            data_speech[int(it[0])] = data[int(it[0])] * detected_windows[it.index,1]
            it.iternext()
        plt.figure()
        plt.plot(data_speech)
        plt.plot(data)
        plt.show()
        return self

    def detect_speech(self):
        """
        Detects speech regions based on ratio between speech band energy
        and total energy.
        Output is array of window numbers and speech flags (1 - speech, 0 - nonspeech).
        """
        detected_windows = np.array([])
        sample_window = int(self.rate * self.sample_window)
        sample_overlap = int(self.rate * self.sample_overlap)
        data = self.data
        sample_start = 0
        start_band = self.speech_start_band
        end_band = self.speech_end_band
        while (sample_start < (len(data) - sample_window)):
            sample_end = sample_start + sample_window
            if sample_end>=len(data): sample_end = len(data)-1
            data_window = data[sample_start:sample_end]
            energy_freq = self._calculate_normalized_energy(data_window)
            sum_voice_energy = self._sum_energy_in_band(energy_freq, start_band, end_band)
            sum_full_energy = sum(energy_freq.values())
            speech_ratio = sum_voice_energy/sum_full_energy
            # Hipothesis is that when there is a speech sequence we have ratio of energies more than Threshold
            speech_ratio = speech_ratio>self.speech_energy_threshold
            detected_windows = np.append(detected_windows,[sample_start, speech_ratio])
            sample_start += sample_overlap
        detected_windows = detected_windows.reshape(int(len(detected_windows)/2),2)
        detected_windows[:,1] = self._smooth_speech_detection(detected_windows)
        return detected_windows



import wave
import webrtcvad
import collections
class VAD_webrtc(object):
    """
    Use webrtcvad package to detect voice activity in wav file
    """
    def __init__(self, wav_file):
        self.data, self.sample_rate = self._read_wav(wav_file)
        self.mode = 3
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(self.mode)
        self.frame_time = 0.02  #20 ms
        self.window_time = 0.2  #200 ms

    def _read_wav(self, wav_file):
        """
        Read .wav file.
        only support 1 channel, 16 bit audio with
        specific sample rate
        """
        wf = wave.open(wav_file, 'rb')
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
        wf.close()

        return pcm_data, sample_rate

    def _frame_split(self, data, frame_time, sample_rate):
        """
        Split PCM audio data to frames.
        """
        # only support 16 bit audio data
        sample_width = 2
        frame_bytes = int(sample_rate * frame_time * sample_width)

        offset = 0
        frame_list = []
        while offset + frame_bytes < len(data):
            frame_list.append(data[offset : offset+frame_bytes])
            offset += frame_bytes

        return frame_list

    def _smooth_speech_detection(self, frames, vad, frame_time, window_time, sample_rate):
        num_window_frames = int(window_time / frame_time)
        # use a deque for sliding window
        sliding_window = collections.deque(maxlen=num_window_frames)
        # flag to indicate smoothed speech detection
        triggered = False

        detected_windows = []
        for frame in frames:
            # raw detection from webrtcvad
            is_speech = vad.is_speech(frame, sample_rate)

            if not triggered:
                sliding_window.append(is_speech)
                # count voiced frames in sliding window
                num_voiced = len([speech for speech in sliding_window if speech])
                # If we're NOTTRIGGERED and more than 90% of the frames in
                # sliding window are voiced frames, then enter the
                # TRIGGERED state.
                if num_voiced > 0.9 * sliding_window.maxlen:
                    triggered = True
                    # clear sliding window
                    sliding_window.clear()
            else:
                sliding_window.append(is_speech)
                # count unvoiced frames in sliding window
                num_unvoiced = len([speech for speech in sliding_window if not speech])
                # If more than 90% of the frames in sliding window are
                # unvoiced, then enter NOTTRIGGERED
                if num_unvoiced > 0.9 * sliding_window.maxlen:
                    triggered = False
                    # clear sliding window
                    sliding_window.clear()

            # record smoothed speech detections for every frame
            # TODO: here we simply drop beginning of the triggered window
            detected_windows.append(int(triggered))

        return detected_windows


    def convert_windows_to_readable_labels(self, detected_windows):
        """
        Take s as input array of window numbers and speech flags from speech
        detection and convert speech flags to time intervals of speech.
        Output is array of dictionaries with speech intervals.
        """
        speech_time = []
        is_speech = 0
        for i, window in enumerate(detected_windows):
            if (window == 1 and is_speech == 0):
                is_speech = 1
                speech_label = {}
                speech_start_time = i * self.frame_time
                speech_label['speech_begin'] = speech_start_time
                #speech_time.append(speech_label)
            if (window == 0 and is_speech == 1):
                is_speech = 0
                speech_end_time = i * self.frame_time
                speech_label['speech_end'] = speech_end_time
                speech_time.append(speech_label)
        return speech_time


    def detect_speech(self):
        frames = self._frame_split(self.data, self.frame_time, self.sample_rate)
        detected_windows = self._smooth_speech_detection(frames, self.vad, self.frame_time, self.window_time, self.sample_rate)
        return detected_windows


def speech_detect(wav_file, vad_type):
    # create VAD object to detect speech
    if vad_type == 'webrtc':
        v = VAD_webrtc(wav_file)
    elif vad_type == 'simple':
        v = VoiceActivityDetector(wav_file)
    else:
        raise ValueError('Unsupported VAD type')

    raw_detection = v.detect_speech()
    speech_labels = v.convert_windows_to_readable_labels(raw_detection)
    #v.plot_detected_speech_regions()

    return speech_labels


def main():
    parser = argparse.ArgumentParser(description='split wakeword section from speech command wav audios, using VAD algorithm')
    parser.add_argument('--wav_path', type=str, required=True,
                        help='input path for wav audios to split')
    parser.add_argument('--split_output_path', type=str, required=True,
                        help='output path for splited wav files')
    parser.add_argument('--backup_path', type=str, required=True,
                        help='path to backup split failed wav files')
    parser.add_argument('--vad_type', type=str, required=False, default='webrtc', choices=['webrtc', 'simple'],
                        help='VAD algorithm type. default=%(default)s')

    args = parser.parse_args()


    # get wav audio file list or single audio
    if os.path.isfile(args.wav_path):
        speech_labels = speech_detect(args.wav_path, args.vad_type)
        print('speech sections: {}'.format(speech_labels))
    else:
        wav_files = glob.glob(os.path.join(args.wav_path, '*.wav'))

        os.makedirs(args.split_output_path, exist_ok=True)
        os.makedirs(args.backup_path, exist_ok=True)

        split_count = 0
        pbar = tqdm(total=len(wav_files), desc='split speech command')
        for wav_file in wav_files:
            # get speech sections for every wav file
            speech_labels = speech_detect(wav_file, args.vad_type)

            if len(speech_labels) == 2:
                split_count += 1
                # found wakeword & speech command, strip
                command_begin_time = speech_labels[1]['speech_begin']
                command_end_time = speech_labels[1]['speech_end']

                sample_rate, data = wf.read(wav_file)
                # get speech command section, and keep some silent at head & tail
                command_begin_sample = int(sample_rate * (command_begin_time - 1.0))
                command_end_sample = int(sample_rate * (command_end_time + 0.5))

                if command_begin_sample < 0:
                    command_begin_sample = 0
                if command_end_sample > len(data):
                    command_end_sample = len(data) - 100

                data = data[command_begin_sample:command_end_sample]

                # save speech command audio
                output_file = os.path.join(args.split_output_path, os.path.basename(wav_file))
                wf.write(output_file, sample_rate, data)

            else:
                # fail to detect speech command, copy to backup path directly
                copy(wav_file, args.backup_path)
            pbar.update(1)
        pbar.close()

        print('number of split wav file:', split_count)


if __name__ == "__main__":
    main()
