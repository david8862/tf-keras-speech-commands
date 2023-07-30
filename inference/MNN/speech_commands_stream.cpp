//
//  speech_commands_stream.cpp
//  MNN
//
//  Created by david8862 on 2023/04/19.
//
#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <getopt.h>
#include <sys/time.h>
#include <assert.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#define MNN_OPEN_TIME_TRACE
#include "MNN/ImageProcess.hpp"
#include "MNN/Interpreter.hpp"
#include "MNN/AutoTime.hpp"
#include "MNN/ErrorCode.hpp"

#include "speech_commands.h"
#include "AudioFile.h"
#include "mfcc.h"
#include "threshold_decoder.h"

using namespace MNN;


void update_audio_buffer(const AudioFile<float> &wav_file, int index, std::vector<float> &audio_buffer, int chunk_size, ListenerParams &listener_params)
{
    // append the chunk audio data to audio buffer
    for (int i = index; i < index+chunk_size; i++) {
        float sample = wav_file.samples[0][i];
        audio_buffer.emplace_back(sample);
    }

    // check audio buffer size, and dequeue head part if need
    if (audio_buffer.size() > listener_params.max_samples()) {
        int dequeue_length = audio_buffer.size() - listener_params.max_samples();
        audio_buffer.erase(audio_buffer.begin(), audio_buffer.begin() + dequeue_length);
    }

    // check feature vectors shape to align with model input
    assert(audio_buffer.size() <= listener_params.max_samples());

    return;
}


void RunInference(Settings* s) {

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

    // assume input tensor type is float
    MNN_ASSERT(feature_input->getType().code == halide_type_float);

    auto shape = feature_input->shape();
    shape[0] = 1;
    net->resizeTensor(feature_input, shape);
    net->resizeSession(session);

    // since we don't need to create other sessions any more,
    // just release model data to save memory
    net->releaseModel();

    // create a host tensor for input data
    auto dataTensor = new Tensor(feature_input, Tensor::TENSORFLOW);

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

    // create host tensor for output data
    std::shared_ptr<Tensor> output_tensor(new Tensor(class_output, class_dim_type));

    // Now we only support float32 type output tensor
    MNN_ASSERT(output_tensor->getType().code == halide_type_float);
    MNN_ASSERT(output_tensor->getType().bits == 32);


    // load wav file with: https://github.com/adamstark/AudioFile
    // which just return the normalized float audio samples
    AudioFile<float> wav_file;
    wav_file.load(s->input_wav_name);

    // show wav file info
    MNN_PRINT("\nInput audio info:\n");
    wav_file.printSummary();

    // check wav file format
    check_wav_file(wav_file, listener_params);


    // initialize input audio buffer
    std::vector<float> audio_buffer(listener_params.max_samples(), 0);

    // initialize feature vector buffer
    int feature_num = listener_params.n_features();
    int feature_size = listener_params.feature_size();
    std::vector<std::vector<float>> feature_vectors(feature_num, std::vector<float>(feature_size, 0));

    // prepare threshold decoder for post process
    ThresholdDecoder threshold_decoder(listener_params.threshold_config, listener_params.threshold_center);


    // loop to listen the wav file
    for (int i = 0; i <= wav_file.getNumSamplesPerChannel() - s->chunk_size; i += s->chunk_size) {
        // read audio data from wav file to audio buffer
        update_audio_buffer(wav_file, i, audio_buffer, s->chunk_size, listener_params);

        // here we pause the loop for some time to simulate real world listen
        usleep(s->chunk_size * 1e6 / listener_params.sample_rate);


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
        fill_data(dataTensor->host<float>(), feature_vectors, s);

        // run speech_commands model
        feature_input->copyFromHostTensor(dataTensor);
        if (net->runSession(session) != NO_ERROR) {
            MNN_PRINT("Failed to invoke MNN!\n");
        }

        // Copy output tensors to host, for further postprocess
        class_output->copyToHostTensor(output_tensor.get());

        std::vector<std::pair<uint8_t, float>> class_results;
        // do speech_commands_postprocess to get sorted command index & scores
        speech_commands_postprocess(output_tensor.get(), class_results);

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

        // detect activations
        bool activate = trigger_detect(classes, index, conf, s->chunk_size, s->conf_thrd, s->trigger_level);

        print_bar(class_name, conf, s->conf_thrd, activate, s->verbose);
    }

    delete dataTensor;
    // Release session and model
    net->releaseSession(session);
    //net->releaseModel();

    MNN_PRINT("\ndone\n");
    return;
}



void display_usage() {
    std::cout
        << "Usage: speech_commands_stream\n"
        << "--mnn_model, -m: model_name.mnn\n"
        << "--params_file, -p: params.json\n"
        << "--classes, -l: classes labels for the model\n"
        << "--wav_file, -i: test.wav\n"
        << "--chunk_size, -c: audio samples between inferences\n"
        << "--sensitivity, -s: model output required to be considered activated\n"
        << "--trigger_level, -g: number of activated chunks to cause an activation\n"
        << "--fast_feature, -e: [0|1], use fast feature extraction or not\n"
        << "--threads, -t: number of threads\n"
        << "--allow_fp16, -f: [0|1], allow running fp32 models with fp16 or not\n"
        << "--verbose, -v: [0|1] print more information\n"
        << "\n";
}


int main(int argc, char** argv)
{
    Settings s;

    int c;
    while (1) {
        static struct option long_options[] = {
            {"mnn_model", required_argument, nullptr, 'm'},
            {"params_file", required_argument, nullptr, 'p'},
            {"classes", required_argument, nullptr, 'l'},
            {"wav_file", required_argument, nullptr, 'i'},
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
                        "c:e:f:g:i:hl:m:p:s:t:v:", long_options,
                        &option_index);

        /* Detect the end of the options. */
        if (c == -1) break;

        switch (c) {
            case 'c':
                s.chunk_size = strtol(  // NOLINT(runtime/deprecated_fn)
                        optarg, nullptr, 10);
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
            case 'i':
                s.input_wav_name = optarg;
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

