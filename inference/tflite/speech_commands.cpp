//
//  speech_commands.cpp
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

#include "speech_commands.h"
#include "AudioFile.h"


namespace speech_commands {

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


    // load wav file with: https://github.com/adamstark/AudioFile
    // which just return the normalized float audio samples
    AudioFile<float> wav_file;
    wav_file.load(s->input_wav_name);

    // show wav file info
    LOG(INFO) << "\nInput audio info:\n";
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
        LOG(INFO) << "\nfirst 10 samples of input audio:\n";
        for (int i = 0; i < 10; i++) {
            LOG(INFO) << audio_buffer[i] << ", ";
        }
        LOG(INFO) << "\n";
    }

    // get frequency domain feature vectors
    gettimeofday(&start_time, nullptr);
    std::vector<std::vector<float>> feature_vectors;
    vectorize(feature_vectors, audio_buffer, listener_params);
    gettimeofday(&stop_time, nullptr);
    LOG(INFO) << "feature vectors extraction time:" << (get_us(stop_time) - get_us(start_time)) / 1000 << " ms\n";

    if (s->verbose) {
        // print feature vectors for check
        LOG(INFO) << "\n feature vectors for input audio:\n";
        for (int i = 0; i < feature_vectors.size(); i++) {
            for (int j = 0; j < feature_vectors[i].size(); j++) {
                LOG(INFO) << feature_vectors[i][j] << ", ";
            }
            LOG(INFO) << "\n";
        }
    }

    // fulfill feature vectors data to model input tensor
    assert(interpreter->tensor(input)->type == kTfLiteFloat32);
    fill_data(interpreter->typed_tensor<float>(input), feature_vectors, s);


    // run warm up session
    if (s->loop_count > 1)
        for (int i = 0; i < s->number_of_warmup_runs; i++) {
            if (interpreter->Invoke() != kTfLiteOk) {
                LOG(FATAL) << "Failed to invoke tflite!\n";
            }
        }

    // run model sessions to get output
    gettimeofday(&start_time, nullptr);
    for (int i = 0; i < s->loop_count; i++) {
        if (interpreter->Invoke() != kTfLiteOk) {
            LOG(FATAL) << "Failed to invoke tflite!\n";
        }
    }
    gettimeofday(&stop_time, nullptr);
    LOG(INFO) << "invoked average time:" << (get_us(stop_time) - get_us(start_time)) / (s->loop_count * 1000) << " ms\n";

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

    std::vector<std::pair<uint8_t, float>> class_results;
    // Do speech_commands_postprocess to get sorted command index & scores
    gettimeofday(&start_time, nullptr);
    speech_commands_postprocess(score_output, class_results);
    gettimeofday(&stop_time, nullptr);
    LOG(INFO) << "speech_commands_postprocess time: " << (get_us(stop_time) - get_us(start_time)) / 1000 << " ms\n";

    // check class size and top_k
    assert(num_classes == class_results.size());
    assert(s->top_k <= num_classes);

    // Open result txt file
    std::ofstream resultOs(s->result_file_name.c_str());

    // Show classification result
    LOG(INFO) << "Inferenced class:\n";
    for(int i = 0; i < s->top_k; i++) {
        auto class_result = class_results[i];
        LOG(INFO) << classes[class_result.first] << ": " << class_result.second << "\n";
        resultOs << classes[class_result.first] << ": " << class_result.second << "\n";
    }

    // release resouces
    resultOs.close();
    return;
}

void display_usage() {
    LOG(INFO)
        << "Usage: speech_commands\n"
        << "--tflite_model, -m: model_name.tflite\n"
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

}  // namespace speech_commands

int main(int argc, char** argv) {
    return speech_commands::Main(argc, argv);
}
