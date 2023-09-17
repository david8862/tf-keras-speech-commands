//
//  speech_commands_stream.cpp
//  Tensorflow-lite
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

#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/string_util.h"

#include "speech_commands.h"
#include "AudioFile.h"
#include "mfcc.h"
#include "threshold_decoder.h"


namespace speech_commands {


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


    // load wav file with: https://github.com/adamstark/AudioFile
    // which just return the normalized float audio samples
    AudioFile<float> wav_file;
    if (!wav_file.load(s->input_wav_name)) {
        LOG(INFO) << "Unable to open wav file!\n";
        exit(-1);
    }

    // show wav file info
    LOG(INFO) << "\nInput audio info:\n";
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

        // detect activations
        bool activate = trigger_detect(classes, index, conf, s->chunk_size, s->conf_thrd, s->trigger_level);

        // print confidence bar
        print_bar(class_name, conf, s->conf_thrd, activate, s->verbose);
    }

    LOG(INFO) << "\ndone\n";
    return;
}


void display_usage() {
    LOG(INFO)
        << "Usage: speech_commands_stream\n"
        << "--tflite_model, -m: model_name.tflite\n"
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


int Main(int argc, char** argv)
{
    Settings s;

    int c;
    while (1) {
        static struct option long_options[] = {
            {"tflite_model", required_argument, nullptr, 'm'},
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

}  // namespace speech_commands

int main(int argc, char** argv) {
    return speech_commands::Main(argc, argv);
}
