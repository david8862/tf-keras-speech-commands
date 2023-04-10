//
//  speech_commands_alsa.cpp
//  Tensorflow-lite
//
//  Created by david8862 on 2022/11/28.
//
#include <stdio.h>
#include <math.h>
#include <getopt.h>
#include <sys/time.h>
#include <assert.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/string_util.h"

#include <alsa/asoundlib.h>
#include "speech_commands.h"
#include "mfcc.h"
#include "threshold_decoder.h"


namespace speech_commands {


#define CHANNEL_NUM 1  // only use single channel for alsa config
#define MAX_BUFFER_TIME 500000  // max alsa PCM buffer time (in us): 500000(0.5s)
void prepare_alsa_device(const std::string &alsa_device, snd_pcm_t* &handle, int chunk_size, ListenerParams &listener_params)
{
    int ret = 0;

    // open alsa device in capture mode
    ret = snd_pcm_open(&handle, alsa_device.c_str(), SND_PCM_STREAM_CAPTURE, 0);
    if (ret < 0) {
        LOG(ERROR) << "Unable to open pcm device!\n";
        exit(-1);
    }

    // create hw param structure
    snd_pcm_hw_params_t* params;
    snd_pcm_hw_params_malloc(&params);

    // init hw params with alsa device
    ret = snd_pcm_hw_params_any(handle, params);
    if (ret < 0) {
        LOG(ERROR) << "Can not configure this PCM device!\n";
        exit(-1);
    }

    // set data arrange type in audio stream
    ret = snd_pcm_hw_params_set_access(handle, params, SND_PCM_ACCESS_RW_INTERLEAVED);
    if (ret < 0) {
        LOG(ERROR) << "Failed to set PCM device to interleaved!\n";
        exit(-1);
    }

    // set sample bits, acturally speech_commands only
    // support int16 format
    assert(listener_params.sample_depth == 2);
    snd_pcm_format_t pcm_format = SND_PCM_FORMAT_S16_LE;
    ret = snd_pcm_hw_params_set_format(handle, params, pcm_format);
    if (ret < 0) {
        LOG(ERROR) << "Failed to set PCM device to 16-bit signed PCM\n";
        exit(-1);
    }

    // set channel number
    ret = snd_pcm_hw_params_set_channels(handle, params, CHANNEL_NUM);
    if (ret < 0) {
        LOG(ERROR) << "Failed to set PCM device channels\n";
        exit(-1);
    }

    // set sample rate
    unsigned int val = listener_params.sample_rate;
    int dir = 0;
    ret = snd_pcm_hw_params_set_rate(handle, params, val, dir);
    if (ret < 0) {
        LOG(ERROR) << "Failed to set PCM device to sample rate\n";
        exit(-1);
    }

    // get max buffer time (in us) and try to set to MAX_BUFFER_TIME
    unsigned int buffer_time;
    snd_pcm_hw_params_get_buffer_time_max(params, &buffer_time, 0);
    if (buffer_time > MAX_BUFFER_TIME) {
        buffer_time = MAX_BUFFER_TIME;
    }

    ret = snd_pcm_hw_params_set_buffer_time_near(handle, params, &buffer_time, 0);
    if (ret < 0) {
        LOG(ERROR) << "Failed to set PCM device to buffer time\n";
        exit(-1);
    }
    LOG(INFO) << "PCM buffer time: " << buffer_time << "\n";

    // set period time
    //unsigned int period_time = 64000;  // 0.1s
    //ret = snd_pcm_hw_params_set_period_time_near(handle, params, &period_time, 0);
    //if (ret < 0) {
        //LOG(ERROR) << "Failed to set PCM device to period time\n";
        //exit(-1);
    //}
    //LOG(INFO) << "PCM period time: " << period_time << "\n";

    // try to set period size to chunk size
    snd_pcm_uframes_t period_size = chunk_size;
    ret = snd_pcm_hw_params_set_period_size_near(handle, params, &period_size, 0);
    if (ret < 0) {
        LOG(ERROR) << "Failed to set PCM device to period size\n";
        exit(-1);
    }
    LOG(INFO) << "PCM sample period size: " << period_size << "\n";

    // apply params to alsa PCM device
    ret = snd_pcm_hw_params(handle, params);
    if (ret < 0) {
        LOG(ERROR) << "Unable to set hw parameters\n";
        exit(-1);
    }

    return;
}


// convert an int16 little endian audio sample from audio buffer
static inline int16_t S16_LE_to_int16(uint8_t* buffer, int index)
{
    int16_t val = (buffer[index + 1] << 8) | buffer[index];

    return val;
}


// convert int16 audio sample to normalized float audio sample
static inline float int16_to_sample(int16_t sample)
{
    return static_cast<float> (sample) / static_cast<float> (32768.);
}


void update_audio_buffer(snd_pcm_t* &handle, std::vector<float> &audio_buffer, int chunk_size, ListenerParams &listener_params)
{
    int ret = 0;
    // here "frame" means 1 sample for all channels,
    // frame_size = sample_size * channels
    int frame_size = listener_params.sample_depth * CHANNEL_NUM;

    int buffer_size = chunk_size * frame_size; // create buffer to read 1 chunk of data
    //LOG(INFO) << "buffer size: " << buffer_size << "\n";

    // create read buffer
    unsigned char* buffer = (unsigned char*)malloc(buffer_size);

    // read 1 chunk of audio data
    ret = snd_pcm_readi(handle, buffer, chunk_size);

    if (ret == -EPIPE) {
        // EPIPE means overrun
        LOG(ERROR) << "Overrun occurred\n";
        ret = snd_pcm_prepare(handle);
        if(ret < 0) {
            LOG(ERROR) << "Failed to recover from overrun\n";
            exit(-1);
        }
    }
    else if (ret < 0) {
        LOG(ERROR) << "error from read: " << snd_strerror(ret) << "\n";
        exit(-1);
    }
    else if (ret != (int)chunk_size) {
        LOG(ERROR) << "short read, read " << ret << " frames\n";
    }

    // append the chunk audio data to audio buffer
    for (int i = 0; i < buffer_size; i += 2) {
        int16_t int16_val = S16_LE_to_int16(buffer, i);
        float sample = int16_to_sample(int16_val);
        audio_buffer.emplace_back(sample);
    }

    // check audio buffer size, and dequeue head part if need
    if (audio_buffer.size() > listener_params.max_samples()) {
        int dequeue_length = audio_buffer.size() - listener_params.max_samples();
        audio_buffer.erase(audio_buffer.begin(), audio_buffer.begin() + dequeue_length);
    }

    // check feature vectors shape to align with model input
    assert(audio_buffer.size() <= listener_params.max_samples());

    free(buffer);
    return;
}


// A more effective approach to update feature vectors, which only need to enqueue
// latest 2, dequeue first 2 features and keep other part unchange, but have following
// limitation:
//
//  1. window_t == 2 * hop_t
//  2. window_t == float(chunk_size) / float(sample_rate)
//
// A typical config:
//
//    sample_rate = 16000
//    chunk_size = 1600
//    window_t = 0.064s
//    hop_t = 0.032s
//
// This may be useful when running speech_commands on some low end CPU, which cost too much
// time in "vectorize" process.
void update_feature_vectors(std::vector<std::vector<float>> &feature_vectors, const std::vector<float> &audio_buffer, ListenerParams &listener_params, int chunk_size)
{
    int sample_rate = listener_params.sample_rate;

    int length_frame = listener_params.window_samples();
    int stride = listener_params.hop_samples();
    int length_FFT = listener_params.n_fft;

    int num_coeffs = listener_params.n_mfcc;
    int num_filters = listener_params.n_filt;

    float window_t = listener_params.window_t;
    float hop_t = listener_params.hop_t;

    // check if params match usage limitation
    if ((window_t != 2 * hop_t) || (window_t != (float(chunk_size)/float(sample_rate)))) {
        LOG(ERROR) << "speech_commands model config doesn't support fast feature, pls double check\n";
        exit(-1);
    }
    //assert(window_t == 2 * hop_t);
    //assert(window_t == (float(chunk_size) / float(sample_rate)));

    // follow frequency config in "sonopy" python package,
    // which use 0 as low & sample_rate as high
    int low_freq = 0;
    int high_freq = sample_rate;

    // no preprocess & delta2
    bool use_preprocess = false;
    bool use_delta = listener_params.use_delta;
    bool use_delta2 = false;

    // assign feature size according to mfcc config
    int feature_size;
    if (use_delta && use_delta2) {
        feature_size = 3 * num_coeffs;
    } else if (use_delta) {
        feature_size = 2 * num_coeffs;
    } else {
        feature_size = num_coeffs;
    }

    // get MFCC feature for the last 1.5 frames of audio buffer
    for (int i = (audio_buffer.size() - (length_frame + stride)); i <= audio_buffer.size() - length_frame; i += stride) {
        std::vector<float> frame(length_frame, 0);
        std::vector<float> feature_vector(feature_size, 0);

        if (use_preprocess) {
            // pre-emphasis
            float alpha = 0.95; // 0.97
            for (int j = 0; j < length_frame; j++) {
                if (i + j < audio_buffer.size()) {
                    frame[j] = audio_buffer[i + j] - alpha * audio_buffer[i + j - 1];
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

                if (i + j < audio_buffer.size()) {
                    frame[j] = audio_buffer[i + j];
                } else {
                    frame[j] = 0;
                }
            }
        }

        // get base MFCC feature vector for 1 frame
        mfcc::mfcc_feature<float>(feature_vector, frame, sample_rate, length_frame, length_FFT, num_coeffs, num_filters, low_freq, high_freq);

        // append to 2-D feature vectors
        feature_vectors.emplace_back(feature_vector);
    }

    // check feature vectors size, and dequeue head part if need
    if (feature_vectors.size() > listener_params.n_features()) {
        int dequeue_length = feature_vectors.size() - listener_params.n_features();
        feature_vectors.erase(feature_vectors.begin(), feature_vectors.begin() + dequeue_length);
    }

    // check feature vectors shape to align with model input
    assert(feature_vectors.size() == listener_params.n_features());

    return;
}


// show confidence bar according to decoded score and threshold
void print_bar(std::string class_name, float confidence, const float threshold)
{
    int total_length = 80;
    std::string score_bar;

    // for background prediction, just show inversed score
    // and ignore label display
    if (class_name == "background") {
        confidence = 1.0 - confidence;
        class_name = "";
    }

    if (confidence > threshold) {
        score_bar = std::string(int(threshold * total_length), 'X') + std::string(int((confidence - threshold) * total_length), 'x');
    }
    else {
        score_bar = std::string(int(confidence * total_length), 'X');
    }

    std::string total_bar = score_bar + std::string(total_length - score_bar.size(), '-') + class_name;
    LOG(INFO) << total_bar << "\n";

    return;
}


// read predictions and detects activations
// align with TriggerDetector in listen.py
bool trigger_detect(const std::vector<std::string> &classes, int index, float conf, int chunk_size, float sensitivity, int trigger_level)
{
    static int activation = 0;
    static int record_index = -1;

    bool chunk_activated = conf > sensitivity;

    if (classes[index] != "background" && index == record_index && chunk_activated) {
        activation += 1;
        bool has_activated = (activation > trigger_level);
        if (has_activated) {
            // reset activation
            activation = int(-(8 * 2048) / chunk_size);
            return true;
        }
    }
    else if (activation < 0) {
        activation += 1;
    }
    else if (activation > 0) {
        activation -= 1;
    }

    // record class index for checking
    record_index = index;
    return false;
}


void RunInference(Settings* s) {
    struct timeval start_time, stop_time;

    if (!s->model_name.c_str()) {
        LOG(ERROR) << "no model file name\n";
        exit(-1);
    }

    // load model
    std::unique_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<tflite::Interpreter> interpreter;
    model = tflite::FlatBufferModel::BuildFromFile(s->model_name.c_str());
    if (!model) {
        LOG(FATAL) << "\nFailed to mmap model " << s->model_name << "\n";
        exit(-1);
    }
    //s->model = model.get();
    LOG(INFO) << "Loaded model " << s->model_name << "\n";
    model->error_reporter();
    LOG(INFO) << "resolved reporter\n";

    // prepare model interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    if (!interpreter) {
        LOG(FATAL) << "Failed to construct interpreter\n";
        exit(-1);
    }

    interpreter->SetAllowFp16PrecisionForFp32(s->allow_fp16);
    if (s->number_of_threads != -1) {
        interpreter->SetNumThreads(s->number_of_threads);
    }

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        LOG(FATAL) << "Failed to allocate tensors!";
    }

    // assuming one input only
    const std::vector<int> inputs = interpreter->inputs();
    assert(inputs.size() == 1);

    // get input dimension from the input tensor metadata
    int input = interpreter->inputs()[0];
    TfLiteIntArray* dims = interpreter->tensor(input)->dims;
    // check input dimension
    assert(dims->size == 3);

    int input_batch = dims->data[0];
    int input_feature_num = dims->data[1];
    int input_feature_size = dims->data[2];

    std::vector<std::string> tensor_type_string = {"kTfLiteNoType",
                                                   "kTfLiteFloat32",
                                                   "kTfLiteInt32",
                                                   "kTfLiteUInt8",
                                                   "kTfLiteInt64",
                                                   "kTfLiteString",
                                                   "kTfLiteBool",
                                                   "kTfLiteInt16",
                                                   "kTfLiteComplex64",
                                                   "kTfLiteInt8",
                                                   "kTfLiteFloat16",
                                                   "kTfLiteFloat64",
                                                   "kTfLiteComplex128",
                                                   "kTfLiteUInt64",
                                                   "kTfLiteResource",
                                                   "kTfLiteVariant",
                                                   "kTfLiteUInt32"
                                                  };

    if (s->verbose) LOG(INFO) << "input tensor info: "
                              << "name " << interpreter->tensor(input)->name << ", "
                              << "type " << tensor_type_string[interpreter->tensor(input)->type] << ", "
                              << "dim_size " << interpreter->tensor(input)->dims->size << ", "
                              << "batch " << input_batch << ", "
                              << "feature_num " << input_feature_num << ", "
                              << "feature_size " << input_feature_size << "\n";

    assert(interpreter->tensor(input)->type == kTfLiteFloat32);


    // get output tensor info, assume only 1 output tensor (Identity)
    // feature_input: 1 x n_features x n_mfcc
    // Identity: 1 x num_classes
    const std::vector<int> outputs = interpreter->outputs();
    assert(outputs.size() == 1);

    int output = interpreter->outputs()[0];
    TfLiteTensor* score_output = interpreter->tensor(output);

    // Now we only support float32 type output tensor
    assert(score_output->type == kTfLiteFloat32);

    TfLiteIntArray* output_dims = score_output->dims;
    // check output dimension
    assert(output_dims->size == 2);
    int output_batch = output_dims->data[0];
    int output_length = output_dims->data[1];

    if (s->verbose) LOG(INFO) << "output tensor info: "
                              << "name " << score_output->name << ", "
                              << "type " << tensor_type_string[score_output->type] << ", "
                              << "dim_size " << score_output->dims->size << ", "
                              << "batch " << output_batch << ", "
                              << "length " << output_length << "\n";


    // model params json config
    ListenerParams listener_params;
    // load & parse params json config to update listener_params
    parse_param(s->params_file_name, listener_params);

    // double check model input shape with updated listener_params
    check_input_shape(input_feature_num, input_feature_size, listener_params);

    // get classes labels
    std::vector<std::string> classes;
    std::ifstream classesOs(s->classes_file_name.c_str());
    std::string line;
    while (std::getline(classesOs, line)) {
        classes.emplace_back(line);
    }
    assert(classes[0] == "background");
    int num_classes = classes.size();
    LOG(INFO) << "num_classes: " << num_classes << "\n";

    // prepare alsa device for audio input
    snd_pcm_t* handle;
    prepare_alsa_device(s->alsa_device, handle, s->chunk_size, listener_params);

    // initialize input audio buffer
    std::vector<float> audio_buffer(listener_params.max_samples(), 0);

    // initialize feature vector buffer
    int feature_num = listener_params.n_features();
    int feature_size = listener_params.feature_size();
    std::vector<std::vector<float>> feature_vectors(feature_num, std::vector<float>(feature_size, 0));

    // prepare threshold decoder for post process
    ThresholdDecoder threshold_decoder(listener_params.threshold_config, listener_params.threshold_center);

    // loop to listen
    while (1) {
        // read audio data from alsa PCM device to audio buffer
        update_audio_buffer(handle, audio_buffer, s->chunk_size, listener_params);

        // update frequency domain feature vectors
        if (s->fast_feature) {
            // use fast approach to update feature vectors
            update_feature_vectors(feature_vectors, audio_buffer, listener_params, s->chunk_size);
        }
        else {
            // standard feature vectorize, update whole feature vectors
            feature_vectors.clear();
            vectorize(feature_vectors, audio_buffer, listener_params);
        }

        // fulfill feature vectors data to model input tensor
        fill_data(interpreter->typed_tensor<float>(input), feature_vectors, s);

        // run speech_commands model
        if (interpreter->Invoke() != kTfLiteOk) {
            LOG(FATAL) << "Failed to invoke tflite!\n";
        }

        std::vector<std::pair<uint8_t, float>> class_results;
        // do speech_commands_postprocess to get sorted command index & scores
        speech_commands_postprocess(score_output, class_results);

        // fetch top command and raw score
        auto class_result = class_results[0];
        int index = class_result.first;
        std::string class_name = classes[index];
        float raw_output = class_result.second;

        float conf;
        // decode non-bg raw score with ThresholdDecoder
        if (class_name != "background") {
            conf = threshold_decoder.decode(raw_output);
        }
        else {
            conf = raw_output;
        }

        // print confidence bar
        print_bar(class_name, conf, s->conf_thrd);

        // detect activations
        bool detected = trigger_detect(classes, index, conf, s->chunk_size, s->conf_thrd, s->trigger_level);
        if (detected) {
            LOG(INFO) << "command " << class_name << " detected!\n";
        }
    }

    // release resouces
    snd_pcm_drain(handle);
    snd_pcm_close(handle);
    return;
}


void display_usage() {
    LOG(INFO)
        << "Usage: speech_commands_alsa\n"
        << "--tflite_model, -m: model_name.tflite\n"
        << "--params_file, -p: params.json\n"
        << "--classes, -l: classes labels for the model\n"
        << "--alsa_device, -d: ALSA device name\n"
        << "--chunk_size, -c: audio samples between inferences\n"
        << "--sensitivity, -s: model output required to be considered activated\n"
        << "--trigger_level, -g: number of activated chunks to cause an activation\n"
        << "--fast_feature, -e: [0|1], use fast feature extraction or not\n"
        << "--threads, -t: number of threads\n"
        << "--allow_fp16, -f: [0|1], allow running fp32 models with fp16 or not\n"
        << "--verbose, -v: [0|1] print more information\n"
        << "\n";
}


int Main(int argc, char** argv)
{
    Settings s;

    int c;
    while (1) {
        static struct option long_options[] = {
            {"tflite_model", required_argument, nullptr, 'm'},
            {"params_file", required_argument, nullptr, 'p'},
            {"classes", required_argument, nullptr, 'l'},
            {"alsa_device", required_argument, nullptr, 'd'},
            {"chunk_size", required_argument, nullptr, 'c'},
            {"sensitivity", required_argument, nullptr, 's'},
            {"trigger_level", required_argument, nullptr, 'g'},
            {"fast_feature", required_argument, nullptr, 'e'},
            {"threads", required_argument, nullptr, 't'},
            {"allow_fp16", required_argument, nullptr, 'f'},
            {"verbose", required_argument, nullptr, 'v'},
            {"help", no_argument, nullptr, 'h'},
            {nullptr, 0, nullptr, 0}};

        /* getopt_long stores the option index here. */
        int option_index = 0;

        c = getopt_long(argc, argv,
                        "c:d:e:f:g:hl:m:p:s:t:v:", long_options,
                        &option_index);

        /* Detect the end of the options. */
        if (c == -1) break;

        switch (c) {
            case 'c':
                s.chunk_size = strtol(  // NOLINT(runtime/deprecated_fn)
                        optarg, nullptr, 10);
                break;
            case 'd':
                s.alsa_device = optarg;
                break;
            case 'e':
                s.fast_feature =
                    strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
                break;
            case 'f':
                s.allow_fp16 =
                    strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
                break;
            case 'g':
                s.trigger_level =
                    strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
                break;
            case 'l':
                s.classes_file_name = optarg;
                break;
            case 'm':
                s.model_name = optarg;
                break;
            case 'p':
                s.params_file_name = optarg;
                break;
            case 's':
                s.conf_thrd = strtod(optarg, nullptr);
                break;
            case 't':
                s.number_of_threads = strtol(  // NOLINT(runtime/deprecated_fn)
                        optarg, nullptr, 10);
                break;
            case 'v':
                s.verbose =
                    strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
                break;
            case 'h':
            case '?':
            default:
                /* getopt_long already printed an error message. */
                display_usage();
                exit(-1);
        }
    }
    RunInference(&s);
    return 0;
}

}  // namespace speech_commands

int main(int argc, char** argv) {
    return speech_commands::Main(argc, argv);
}
