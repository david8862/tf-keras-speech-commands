//
//  speech_commands.cpp
//  MNN
//
//  Created by david8862 on 2022/12/01.
//
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <algorithm>
#include <fstream>
#include <functional>
#include <memory>
#include <sstream>
#include <iostream>
#include <vector>
#include <numeric>
#include <math.h>
#include <unistd.h>
#include <getopt.h>
#include <string.h>
#include <sys/time.h>

#define MNN_OPEN_TIME_TRACE
#include "MNN/ImageProcess.hpp"
#include "MNN/Interpreter.hpp"
#include "MNN/AutoTime.hpp"
#include "MNN/ErrorCode.hpp"

#include "speech_commands.h"
#include "AudioFile.h"

using namespace MNN;


void check_wav_file(const AudioFile<float> &wav_file, ListenerParams &listener_params)
{
    int num_channels = wav_file.getNumChannels();
    int sample_rate = wav_file.getSampleRate();
    int bit_depth = wav_file.getBitDepth();

    // input audio format:
    // single channel, 16k, 16bit audio
    assert(num_channels == 1 &&
           sample_rate == listener_params.sample_rate &&
           bit_depth == listener_params.sample_depth*8);

    // double check audio sample size
    assert(wav_file.samples[0].size() == wav_file.getNumSamplesPerChannel());

    return;
}


void RunInference(Settings* s) {
    // record run time for every stage
    struct timeval start_time, stop_time;

    // create model & session
    std::shared_ptr<Interpreter> net(Interpreter::createFromFile(s->model_name.c_str()));
    ScheduleConfig config;
    config.type  = MNN_FORWARD_AUTO; //MNN_FORWARD_CPU, MNN_FORWARD_OPENCL
    config.backupType = MNN_FORWARD_CPU;
    config.numThread = s->number_of_threads;

    BackendConfig bnconfig;
    bnconfig.memory = BackendConfig::Memory_Normal; //Memory_High, Memory_Low
    bnconfig.power = BackendConfig::Power_Normal; //Power_High, Power_Low
    bnconfig.precision = BackendConfig::Precision_Normal; //Precision_High, Precision_Low
    config.backendConfig = &bnconfig;

    auto session = net->createSession(config);

    // get input tensor info, assume only 1 input tensor (feature_input)
    auto inputs = net->getSessionInputAll(session);
    MNN_ASSERT(inputs.size() == 1);
    auto feature_input = inputs.begin()->second;
    int input_width = feature_input->width();
    int input_height = feature_input->height();
    int input_channel = feature_input->channel();
    auto input_dim_type = feature_input->getDimensionType();

    std::vector<std::string> dim_type_string = {"TENSORFLOW", "CAFFE", "CAFFE_C4"};

    MNN_PRINT("feature_input: name:%s, width:%d, height:%d, channel:%d, dim_type:%s\n", inputs.begin()->first.c_str(), input_width, input_height, input_channel, dim_type_string[input_dim_type].c_str());

    auto shape = feature_input->shape();
    shape[0] = 1;
    net->resizeTensor(feature_input, shape);
    net->resizeSession(session);

    // since we don't need to create other sessions any more,
    // just release model data to save memory
    net->releaseModel();

    // model params json config
    ListenerParams listener_params;
    // load & parse params json config to update listener_params
    parse_param(s->params_file_name, listener_params);

    // double check model input shape with updated listener_params
    // for MNN model, input_height is feature num and input_width is
    // feature size
    int input_feature_num = input_height;
    int input_feature_size = input_width;
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
    MNN_PRINT("num_classes: %d\n", num_classes);

    // load wav file with: https://github.com/adamstark/AudioFile
    // which just return the normalized float audio samples
    AudioFile<float> wav_file;
    wav_file.load(s->input_wav_name);

    // show wav file info
    MNN_PRINT("\nInput audio info:\n");
    wav_file.printSummary();

    // check wav file format
    check_wav_file(wav_file, listener_params);

    // create input audio buffer
    std::vector<float> audio_buffer(listener_params.max_samples(), 0);

    if (wav_file.getNumSamplesPerChannel() <= listener_params.max_samples()) {
        // audio file is short than input buffer,
        // copy all samples to tail of input buffer
        int index_shift = listener_params.max_samples() - wav_file.getNumSamplesPerChannel();
        for (int i = 0; i < wav_file.getNumSamplesPerChannel(); i++) {
            audio_buffer[i + index_shift] = wav_file.samples[0][i];
            //audio_buffer[i] = wav_file.samples[0][i];
        }
    } else {
        // audio file is longer than input buffer,
        // just copy tail part to align with vectorization.py
        int index_shift = wav_file.getNumSamplesPerChannel() - listener_params.max_samples();

        for (int i = 0; i < audio_buffer.size(); i++) {
            audio_buffer[i] = wav_file.samples[0][i + index_shift];
            //audio_buffer[i] = wav_file.samples[0][i];
        }
    }

    if (s->verbose) {
        MNN_PRINT("\nfirst 10 samples of input audio:\n");
        for (int i = 0; i < 10; i++) {
            MNN_PRINT("%f, ", audio_buffer[i]);
        }
        MNN_PRINT("\n");
    }

    // get frequency domain feature vectors
    gettimeofday(&start_time, nullptr);
    std::vector<std::vector<float>> feature_vectors;
    vectorize(feature_vectors, audio_buffer, listener_params);
    gettimeofday(&stop_time, nullptr);
    MNN_PRINT("feature vectors extraction time: %lf ms\n", (get_us(stop_time) - get_us(start_time)) / 1000);

    if (s->verbose) {
        // print feature vectors for check
        MNN_PRINT("\n feature vectors for input audio:\n");
        for (int i = 0; i < feature_vectors.size(); i++) {
            for (int j = 0; j < feature_vectors[i].size(); j++) {
                MNN_PRINT("%f, ", feature_vectors[i][j]);
            }
            MNN_PRINT("\n");
        }
    }

    // assume input tensor type is float
    MNN_ASSERT(feature_input->getType().code == halide_type_float);

    // create a host tensor for input data
    auto dataTensor = new Tensor(feature_input, Tensor::TENSORFLOW);
    fill_data(dataTensor->host<float>(), feature_vectors, s);

    // run warm up session
    if (s->loop_count > 1)
        for (int i = 0; i < s->number_of_warmup_runs; i++) {
            feature_input->copyFromHostTensor(dataTensor);
            if (net->runSession(session) != NO_ERROR) {
                MNN_PRINT("Failed to invoke MNN!\n");
            }
        }

    // run model sessions to get output
    gettimeofday(&start_time, nullptr);
    for (int i = 0; i < s->loop_count; i++) {
        feature_input->copyFromHostTensor(dataTensor);
        if (net->runSession(session) != NO_ERROR) {
            MNN_PRINT("Failed to invoke MNN!\n");
        }
    }
    gettimeofday(&stop_time, nullptr);
    MNN_PRINT("model invoke average time: %lf ms\n", (get_us(stop_time) - get_us(start_time)) / (1000 * s->loop_count));


    // get output tensor info, assume only 1 output tensor (Identity)
    // feature_input: 1 x n_features x n_mfcc
    // Identity: 1 x num_classes
    auto outputs = net->getSessionOutputAll(session);
    MNN_ASSERT(outputs.size() == 1);

    auto class_output = outputs.begin()->second;
    int class_width = class_output->width();
    int class_height = class_output->height();
    int class_channel = class_output->channel();
    auto class_dim_type = class_output->getDimensionType();
    MNN_PRINT("output tensor: name:%s, width:%d, height:%d, channel:%d, dim_type:%s\n", outputs.begin()->first.c_str(), class_width, class_height, class_channel, dim_type_string[class_dim_type].c_str());

    // get class dimension according to different tensor format
    int class_size;
    if (class_dim_type == Tensor::TENSORFLOW) {
        // Tensorflow format tensor, NHWC
        MNN_PRINT("Tensorflow format: NHWC\n");
        class_size = class_height;
    } else if (class_dim_type == Tensor::CAFFE) {
        // Caffe format tensor, NCHW
        MNN_PRINT("Caffe format: NCHW\n");
        class_size = class_channel;
    } else if (class_dim_type == Tensor::CAFFE_C4) {
        MNN_PRINT("Caffe format: NC4HW4, not supported\n");
        exit(-1);
    } else {
        MNN_PRINT("Invalid tensor dim type: %d\n", class_dim_type);
        exit(-1);
    }

    // check if predict class number matches label file
    MNN_ASSERT(num_classes == class_size);

    // Copy output tensors to host, for further postprocess
    std::shared_ptr<Tensor> output_tensor(new Tensor(class_output, class_dim_type));
    class_output->copyToHostTensor(output_tensor.get());

    // Now we only support float32 type output tensor
    MNN_ASSERT(output_tensor->getType().code == halide_type_float);
    MNN_ASSERT(output_tensor->getType().bits == 32);


    std::vector<std::pair<uint8_t, float>> class_results;
    // Do speech_commands_postprocess to get sorted class index & scores
    gettimeofday(&start_time, nullptr);
    speech_commands_postprocess(output_tensor.get(), class_results);
    gettimeofday(&stop_time, nullptr);
    MNN_PRINT("speech_commands_postprocess time: %lf ms\n", (get_us(stop_time) - get_us(start_time)) / 1000);

    // check class size and top_k
    MNN_ASSERT(num_classes == class_results.size());
    MNN_ASSERT(s->top_k <= num_classes);

    // Open result txt file
    std::ofstream resultOs(s->result_file_name.c_str());

    // Show classification result
    MNN_PRINT("Inferenced class:\n");
    for(int i = 0; i < s->top_k; i++) {
        auto class_result = class_results[i];
        MNN_PRINT("%s: %f\n", classes[class_result.first].c_str(), class_result.second);
        resultOs << classes[class_result.first] << ": " << class_result.second << "\n";
    }

    delete dataTensor;
    // Release session and model
    net->releaseSession(session);
    //net->releaseModel();
    return;
}


void display_usage() {
    std::cout
        << "Usage: speech_commands\n"
        << "--mnn_model, -m: model_name.mnn\n"
        << "--params_file, -p: params.json\n"
        << "--classes, -l: classes labels for the model\n"
        << "--top_k, -k: show top k classes result\n"
        << "--wav_file, -i: test.wav\n"
        << "--allow_fp16, -f: [0|1], allow running fp32 models with fp16 or not\n"
        << "--threads, -t: number of threads\n"
        << "--count, -c: loop interpreter->Invoke() for certain times\n"
        << "--warmup_runs, -w: number of warmup runs\n"
        << "--result, -r: result txt file to save detection output\n"
        << "--verbose, -v: [0|1] print more information\n"
        << "\n";
    return;
}


int main(int argc, char** argv) {
    Settings s;

    int c;
    while (1) {
        static struct option long_options[] = {
            {"mnn_model", required_argument, nullptr, 'm'},
            {"params_file", required_argument, nullptr, 'p'},
            {"classes", required_argument, nullptr, 'l'},
            {"top_k", required_argument, nullptr, 'k'},
            {"wav_file", required_argument, nullptr, 'i'},
            {"threads", required_argument, nullptr, 't'},
            {"allow_fp16", required_argument, nullptr, 'f'},
            {"count", required_argument, nullptr, 'c'},
            {"warmup_runs", required_argument, nullptr, 'w'},
            {"result", required_argument, nullptr, 'r'},
            {"verbose", required_argument, nullptr, 'v'},
            {"help", no_argument, nullptr, 'h'},
            {nullptr, 0, nullptr, 0}};


        /* getopt_long stores the option index here. */
        int option_index = 0;

        c = getopt_long(argc, argv,
                        "c:f:hi:k:l:m:p:r:t:v:w:", long_options,
                        &option_index);

        /* Detect the end of the options. */
        if (c == -1) break;

        switch (c) {
            case 'c':
                s.loop_count =
                    strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
                break;
            case 'f':
                s.allow_fp16 =
                    strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
                break;
            case 'i':
                s.input_wav_name = optarg;
                break;
            case 'k':
                s.top_k=
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
            case 'r':
                s.result_file_name = optarg;
                break;
            case 't':
                s.number_of_threads = strtol(  // NOLINT(runtime/deprecated_fn)
                        optarg, nullptr, 10);
                break;
            case 'v':
                s.verbose =
                    strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
                break;
            case 'w':
                s.number_of_warmup_runs =
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

