// Reference from:
// https://blog.csdn.net/LiuPeiP_VIPL/article/details/81742392
// https://www.cnblogs.com/LXP-Never/p/16011229.html
// https://github.com/MycroftAI/sonopy/blob/master/sonopy.py
#ifndef MFCC_H
#define MFCC_H

#define  _USE_MATH_DEFINES
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <complex>

namespace mfcc {

// Prevents error on log(0) or log(-1)
#define EPSILON 2.220446e-16
static double safe_log(double value)
{
    if (value <= 0) {
        return log(EPSILON);
    } else {
        return log(value);
    }
}

static double safe_log10(double value)
{
    if (value <= 0) {
        return log10(EPSILON);
    } else {
        return log10(value);
    }
}


// Implementation of Discrete Cosine Transform (Type II, norm='ortho'), see
// https://docs.scipy.org/doc/scipy/reference/generated/scipy.fftpack.dct.html
// direction == 1  : DCT
// direction == -1 : inverted DCT
static void DCT(int direction, int length, std::vector<double> &data)
{
    std::vector<double> x(length, 0);

    if (direction != 1 && direction != -1) {
        std::cerr << "invalid DCT direction!" << std::endl;
        return;
    }

    for (int i = 0; i < length; i++) {
        x[i] = data[i];
    }

    for (int k = 0; k < length; k++) {
        double sum = 0;

        if (direction == 1) {
            for (int n = 0; n < length; n++) {
                sum += ((k == 0) ? (sqrt(0.5)) : (1)) * x[n] * cos(M_PI * (n + 0.5) * k / length);
            }
        } else if (direction == -1) {
            for (int n = 0; n < length; n++) {
                sum += ((n == 0) ? (sqrt(0.5)) : (1)) * x[n] * cos(M_PI * n * (k + 0.5) / length);
            }
        }
        data[k] = sum * sqrt(2.0 / length);
    }

    return;
}


// Implementation of Fast Fourier Transform
// direction == 1  : FFT
// direction == -1 : inverted FFT
static void FFT(int direction, int length, std::vector<std::complex<double>> &data)
{
    // convert FFT length to its log2 base, using following function:
    // log2(x) = log(x) / log(2)
    int log_length = (int)(safe_log((double)length) / safe_log(2.0));

    if (direction != 1 && direction != -1) {
        std::cerr << "invalid FFT direction!" << std::endl;
        return;
    }

    // double check if FFT length is 2^n
    if (1 << log_length != length) {
        std::cerr << "invalid FFT length!" << std::endl;
        return;
    }

    for (int i = 0, j = 0; i < length; i++, j = 0) {
        for (int k = 0; k < log_length; k++) {
            j = (j << 1) | (1 & (i >> k));
        }

        if (j < i) {
            std::complex<double> tmp;
            tmp = data[i];
            data[i] = data[j];
            data[j] = tmp;
        }
    }

    for (int i = 0; i < log_length; i++) {
        int L = (int)pow(2.0, i);

        for (int j = 0; j < length - 1; j += 2 * L) {
            for (int k = 0; k < L; k++) {
                double argument = direction * -M_PI * k / L;

                double temp_real = data[j+k+L].real() * cos(argument) - data[j+k+L].imag() * sin(argument);
                double temp_image = data[j+k+L].real() * sin(argument) + data[j+k+L].imag() * cos(argument);
                std::complex<double> temp{temp_real, temp_image};
                data[j+k+L] = data[j+k] - temp;
                data[j+k] = data[j+k] + temp;
            }
        }
    }

    if (direction == -1) {
        for (int k = 0; k < length; k++) {
            data[k] /= length;
        }
    }

    return;
}


// convert frequency to mel scale
static inline double freq_to_mel(double x)
{
    return 1127.0 * safe_log(1 + x / 700.0);
    //return 2595 * log10(1 + x / 700.0);
}

// convert mel scale to frequency
static inline double mel_to_freq(double x)
{
    return 700.0 * (exp(x / 1127.0) - 1);
    //return 700.0 * (pow(10, x / 2595.0) - 1);
}


#if 0
// legacy function. compute mel filterbanks, together with power
static void filterbanks_with_power(std::vector<double> &filterbank, std::vector<std::complex<double>> &fft_points, double &total_power, int length_FFT, int num_filters, int low_freq, int high_freq)
{
    double low_freq_mel = freq_to_mel(low_freq);
    double high_freq_mel = freq_to_mel(high_freq);
    double interval = (high_freq_mel - low_freq_mel) / (num_filters + 1);

    for (int i = 0; i < length_FFT / 2 + 1; i++) {
        double frequency = high_freq * i / (length_FFT / 2);
        double mel_freq = freq_to_mel(frequency);

        // Collect energy value
        double power = std::norm(fft_points[i]) / length_FFT;
        total_power += power;

        // Mel filterbanks
        for (int j = 0; j < num_filters; j++) {
            double frequency_boundary[] = { low_freq_mel + interval * (j + 0), low_freq_mel + interval * (j + 1), low_freq_mel + interval * (j + 2) };

            if (frequency_boundary[0] <= mel_freq && mel_freq <= frequency_boundary[1]) {
                double lower_frequency = mel_to_freq(frequency_boundary[0]);
                double upper_frequency = mel_to_freq(frequency_boundary[1]);

                filterbank[j] += power * (frequency - lower_frequency) / (upper_frequency - lower_frequency);
            } else if (frequency_boundary[1] <= mel_freq && mel_freq <= frequency_boundary[2]) {
                double lower_frequency = mel_to_freq(frequency_boundary[1]);
                double upper_frequency = mel_to_freq(frequency_boundary[2]);

                filterbank[j] += power * (upper_frequency - frequency) / (upper_frequency - lower_frequency);
            }
        }
    }

    return;
}
#endif


// Convert power spectrogram (amplitude squared) to decibel (dB) units,
// using max value of the spectrogram as reference.
// Reference from librosa.power_to_db()
static void power_to_db(std::vector<std::vector<double>> &spec)
{
    // check the input spectrogram is not empty
    assert(!spec.empty());

    // get max value of the spectrogram for reference
    double ref_val = *std::max_element(spec[0].begin(), spec[0].end());
    for (int i = 0; i < spec.size(); i++) {
        double max_val = *std::max_element(spec[i].begin(), spec[i].end());
        ref_val = (max_val > ref_val) ? max_val : ref_val;
    }

    for (int i = 0; i < spec.size(); i++) {
        for (int j = 0; j < spec[i].size(); j++) {
            double val = 10.0 * safe_log10(spec[i][j]) - 10.0 * safe_log10(ref_val);
            spec[i][j] = val;
        }
    }

    return;
}


// calculate power spectrogram and total power
static double power_spec(std::vector<double> &powers, std::vector<std::complex<double>> &fft_points, int length_FFT)
{
    double total_power = 0;
    for (int i = 0; i < length_FFT / 2 + 1; i++) {
        // collect power value & total power
        double power = std::norm(fft_points[i]) / length_FFT;

        powers.emplace_back(power);
        total_power += power;
    }
    return total_power;
}


// compute mel filterbanks
// NOTE: it has nothing to do with the audio data, but only audio & MFCC config
static void filterbanks(std::vector<std::vector<double>> &filterbank, int sample_rate, int length_FFT, int num_filters, int low_freq, int high_freq)
{
    // initial filterbank should be empty
    assert(filterbank.empty());

    double low_freq_mel = freq_to_mel(low_freq);
    double high_freq_mel = freq_to_mel(high_freq);

    std::vector<int> freq_points;
    int point_num = num_filters + 2;
    double step = (high_freq_mel - low_freq_mel) / double(point_num - 1);

    for (int i = 0; i < point_num; i++) {
        double mel_val = low_freq_mel + i * step;
        double freq_val = mel_to_freq(mel_val);
        int point_val = int(freq_val * (length_FFT / 2 + 1) / sample_rate);

        freq_points.emplace_back(point_val);
    }

    for (int i = 0; i < num_filters; i++) {
        std::vector<double> bank(length_FFT / 2 + 1, 0);

        for (int j = freq_points[i]; j < freq_points[i+1]; j++) {
            bank[j] = double(j - freq_points[i]) / double(freq_points[i+1] - freq_points[i]);
        }
        for (int j = freq_points[i+1]; j < freq_points[i+2]; j++) {
            bank[j] = double(freq_points[i+2] - j) / double(freq_points[i+2] - freq_points[i+1]);
        }

        filterbank.emplace_back(bank);
    }

    return;
}


// apply mel filterbanks on power spec to get mel spectogram,
// which is just calculating:
//
//     np.dot(powers, filterbank.T)
//
static void mel_spectogram(std::vector<double> &powers, std::vector<std::vector<double>> &filterbank, std::vector<double> &mel_spec, int num_filters, int length_FFT)
{
    // check input & output shape
    assert(powers.size() == length_FFT / 2 + 1);
    assert(filterbank.size() == num_filters);
    assert(mel_spec.size() == num_filters);

    for (int i = 0; i < num_filters; i++) {
        assert(filterbank[i].size() == length_FFT / 2 + 1);

        double mel_val = 0;
        for (int j = 0; j < length_FFT / 2 + 1; j++) {
            mel_val += powers[j] * filterbank[i][j];
        }
        mel_spec[i] = mel_val;
    }

    return;
}


// get base MFCC feature vector for 1 frame audio data, including:
// FFT -> Mel filterbanks -> log -> DCT
template <typename T>
void mfcc_feature(std::vector<T> &feature_vector, const std::vector<T> &frame, int sample_rate,
                  int length_frame, int length_FFT, int num_coeffs, int num_filters,
                  int low_freq=-1, int high_freq=-1)
{
    static_assert(std::is_floating_point<T>::value, "ERROR: This version of mfcc_feature only supports floating point data formats");

    // check for valid feature params
    assert(num_coeffs <= num_filters);

    // if all frame data is 0, just return 0 feature vector
    bool is_all_zero = std::all_of(std::begin(frame),
                                   std::end(frame),
                                   [](T item) { return item == 0; });

    if (is_all_zero) {
        for (int i = 0; i < num_coeffs; i++) {
            feature_vector[i] = 0.0;
        }
        return;
    }

    // set default low/high frequency value
    if (low_freq == -1) {
        low_freq = 300;
    }
    if (high_freq == -1) {
        high_freq = sample_rate / 2;
    }

    // compute FFT
    std::vector<std::complex<double>> fft_points(length_FFT, std::complex<double>(0, 0));
    for (int i = 0; i < length_FFT; i++) {
        double value = (i < length_frame) ? (frame[i]) : (0);
        fft_points[i].real(value);
    }
    FFT(1, length_FFT, fft_points);

    // compute frequency domain power value
    std::vector<double> powers;
    double total_power = power_spec(powers, fft_points, length_FFT);

    // generate mel filterbanks
    std::vector<std::vector<double>> filterbank;
    filterbanks(filterbank, sample_rate, length_FFT, num_filters, low_freq, high_freq);

    // apply filterbanks to get mel spectogram
    std::vector<double> mel_spec(num_filters, 0);
    mel_spectogram(powers, filterbank, mel_spec, num_filters, length_FFT);

    // log action
    for (int i = 0; i < num_filters; i++) {
        mel_spec[i] = safe_log(mel_spec[i]);
    }

    // compute DCT
    DCT(1, num_filters, mel_spec);

    // Fetch MFCC feature values
    for (int i = 0; i < num_coeffs; i++) {
        feature_vector[i] = mel_spec[i];
    }

    // Replace first band with log energies
    feature_vector[0] = safe_log(total_power);

    return;
}


// calculate MFCC feature for audio data segment
template <typename T>
void mfcc(std::vector<std::vector<T>> &feature_vectors, const std::vector<T> &audio_data, int sample_rate,
          int length_frame, int stride, int length_FFT, int num_coeffs, int num_filters,
          int low_freq=-1, int high_freq=-1,
          bool use_preprocess=false, bool use_delta=false, bool use_delta2=false)
{
    static_assert(std::is_floating_point<T>::value, "ERROR: This version of mfcc only supports floating point data formats");

    // initial feature vectors should be empty
    assert(feature_vectors.empty());

    // assign feature size according to mfcc config
    int feature_size;
    if (use_delta && use_delta2) {
        feature_size = 3 * num_coeffs;
    } else if (use_delta) {
        feature_size = 2 * num_coeffs;
    } else {
        feature_size = num_coeffs;
    }

    int num_feature_vector = (audio_data.size() - length_frame) / stride + 1;

    // get MFCC feature for each frame
    for (int i = 0; i <= audio_data.size() - length_frame; i += stride) {
        std::vector<T> frame(length_frame, 0);
        std::vector<T> feature_vector(feature_size, 0);

        if (use_preprocess) {
            // pre-emphasis
            float alpha = 0.95; // 0.97
            for (int j = 0; j < length_frame; j++) {
                if (i + j < audio_data.size()) {
                    frame[j] = audio_data[i + j] - alpha * audio_data[i + j - 1];
                } else {
                    frame[j] = 0;
                }
            }

            // apply hamming/hanning/rect window
            for (int j = 0; j < length_frame; j++) {
                frame[j] *= 0.54 - 0.46 * cos(2 * M_PI * j / (length_frame - 1)); // hamming
                //frame[j] *= 0.5 - 0.5 * cos(2 * M_PI * j / (length_frame - 1)); // hanning
                //frame[j] *= 1; // rect
            }

        } else {
            // get frame
            for (int j = 0; j < length_frame; j++) {

                if (i + j < audio_data.size()) {
                    frame[j] = audio_data[i + j];
                } else {
                    frame[j] = 0;
                }
            }
        }

        // get base MFCC feature vector for 1 frame
        mfcc_feature<T>(feature_vector, frame, sample_rate, length_frame, length_FFT, num_coeffs, num_filters, low_freq, high_freq);

        // append to 2-D feature vectors
        feature_vectors.emplace_back(feature_vector);
    }

    // calculate deltas
    if (use_delta) {
        for (int i = 0; i < num_feature_vector; i++) {
            int prev = (i == 0) ? (0) : (i - 1);
            int next = (i == num_feature_vector - 1) ? (num_feature_vector - 1) : (i + 1);

            for (int j = 0; j < num_coeffs; j++) {
                feature_vectors[i][num_coeffs + j] = (feature_vectors[next][j] - feature_vectors[prev][j]) / 2;
            }
        }
    }

    // calculate delta-deltas
    if (use_delta2) {
        for (int i = 0; i < num_feature_vector; i++) {
            int prev = (i == 0) ? (0) : (i - 1);
            int next = (i == num_feature_vector - 1) ? (num_feature_vector - 1) : (i + 1);

            for (int j = num_coeffs; j < 2 * num_coeffs; j++) {
                feature_vectors[i][num_coeffs + j] = (feature_vectors[next][j] - feature_vectors[prev][j]) / 2;
            }
        }
    }

    return;
}

}  // namespace mfcc

#endif // MFCC_H
