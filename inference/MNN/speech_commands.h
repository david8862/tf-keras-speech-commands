//
//  speech_commands.h
//  MNN
//
//  Created by david8862 on 2022/11/28.
//
//
#ifndef SPEECH_COMMANDS_H_
#define SPEECH_COMMANDS_H_

#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <cjson/cJSON.h>

#define MNN_OPEN_TIME_TRACE
#include "MNN/ImageProcess.hpp"
#include "MNN/Interpreter.hpp"
#include "MNN/AutoTime.hpp"
#include "MNN/ErrorCode.hpp"

#include "mfcc.h"

#define LOG(x) std::cout

using namespace MNN;


// align with classifier/params.py
class ListenerParams {
public:
    float buffer_t = 1.0;
    float window_t = 0.064;
    float hop_t = 0.032;

    int sample_rate = 16000;
    int sample_depth = 2;

    int n_fft = 1024;
    int n_filt = 20;
    int n_mfcc = 20;

    bool use_delta = false;

    std::vector<float> threshold_config = {6.0, 4.0};
    float threshold_center = 0.2;


    int window_samples(void) {
        // window_t converted to samples
        return int(sample_rate * window_t + 0.5);
    }

    int hop_samples(void) {
        // hop_t converted to samples
        return int(sample_rate * hop_t + 0.5);
    }

    int max_samples(void) {
        // The input size converted to audio samples
        return int(buffer_t * sample_rate);
    }

    int buffer_samples(void) {
        // buffer_t converted to samples, truncating partial frames
        int samples = int(sample_rate * buffer_t + 0.5);
        return hop_samples() * (samples / hop_samples());
    }

    int n_features(void) {
        // Number of timesteps in one input to the network
        return 1 + int(floor((buffer_samples() - window_samples()) / hop_samples()));
    }

    int feature_size(void) {
        // The size of an input vector generated with these parameters
        int num_features = n_mfcc;

        if (use_delta) {
            num_features *= 2;
        }
        return num_features;
    }
};


struct Settings {
    bool verbose = false;
    bool allow_fp16 = false;
    int loop_count = 1;
    int top_k = 1;
    float conf_thrd = 0.5f;
    std::string model_name = "./model.mnn";
    std::string params_file_name = "./params.json";
    std::string classes_file_name = "./classes.txt";
    std::string input_wav_name = "./test.wav";
    std::string result_file_name = "./result.txt";
    std::string alsa_device = "plughw:3,0";
    int chunk_size = 1024;
    int trigger_level = 3;
    bool fast_feature = false;
    int number_of_threads = 4;
    int number_of_warmup_runs = 2;
};


double inline get_us(struct timeval t)
{
    return (t.tv_sec * 1000000 + t.tv_usec);
}


//parse params JSON config file
void parse_param(const std::string &param_file, ListenerParams &listener_params)
{
    int ret;
    //get JSON data from file
    FILE *fp = fopen(param_file.c_str(), "r");
    if ( fp == nullptr ) {
        LOG(ERROR) << "param file open failed\n";
        return;
    }

    // get data size
    fseek(fp, 0, SEEK_END);
    long len = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    char *data = new char [len + 1];
    ret = fread(data, 1, len, fp);
    fclose(fp);

    //parse JSON data
    cJSON *root_json = cJSON_Parse(data);    //parse string to JSON struct
    if (root_json == nullptr) {
        LOG(ERROR) << "error parse root json: " << cJSON_GetErrorPtr() <<"\n";
        cJSON_Delete(root_json);
        delete [] data;
        return;
    }

    // NOTE: we didn't check json content but directly fetch value for simple code.
    //       may have problem for broken json file

    // "buffer_t": 1.0
    listener_params.buffer_t = cJSON_GetObjectItem(root_json, "buffer_t")->valuedouble;
    // "window_t": 0.064
    listener_params.window_t = cJSON_GetObjectItem(root_json, "window_t")->valuedouble;
    // "hop_t": 0.032
    listener_params.hop_t = cJSON_GetObjectItem(root_json, "hop_t")->valuedouble;

    // "sample_rate": 16000
    listener_params.sample_rate = cJSON_GetObjectItem(root_json, "sample_rate")->valueint;
    // "sample_depth": 2
    listener_params.sample_depth = cJSON_GetObjectItem(root_json, "sample_depth")->valueint;

    // "n_fft": 1024
    listener_params.n_fft = cJSON_GetObjectItem(root_json, "n_fft")->valueint;
    // "n_filt": 20
    listener_params.n_filt = cJSON_GetObjectItem(root_json, "n_filt")->valueint;
    // "n_mfcc": 20
    listener_params.n_mfcc = cJSON_GetObjectItem(root_json, "n_mfcc")->valueint;

    // "use_delta": false
    listener_params.use_delta = bool(cJSON_IsTrue(cJSON_GetObjectItem(root_json, "use_delta")));

    // "threshold_config": [[6, 4]], 2-dim array
    // currently we only parse the 1st pair of the config
    cJSON* thr_cfg_json = cJSON_GetObjectItem(root_json, "threshold_config")->child;
    int thr_cfg_json_size = cJSON_GetArraySize(thr_cfg_json);
    //LOG(INFO) << "threshold_config json size:" << thr_cfg_json_size << "\n";
    assert(listener_params.threshold_config.size() == thr_cfg_json_size);
    for (int i = 0; i < thr_cfg_json_size; i++) {
        listener_params.threshold_config[i] = cJSON_GetArrayItem(thr_cfg_json, i)->valuedouble;
    }

    // "threshold_center": 0.2
    listener_params.threshold_center = cJSON_GetObjectItem(root_json, "threshold_center")->valuedouble;

    cJSON_Delete(root_json);
    delete [] data;
    LOG(INFO) << "params json parsed\n";
    return;
}


void check_input_shape(const int input_feature_num, const int input_feature_size, ListenerParams &listener_params)
{
    assert(input_feature_num == listener_params.n_features());
    assert(input_feature_size == listener_params.feature_size());

    return;
}


// convert audio frames to feature vectors, currently only support mfcc feature
void vectorize(std::vector<std::vector<float>> &feature_vectors, const std::vector<float> &audio_buffer, ListenerParams &listener_params)
{
    int sample_rate = listener_params.sample_rate;

    int length_frame = listener_params.window_samples();
    int stride = listener_params.hop_samples();
    int length_FFT = listener_params.n_fft;

    int num_coeffs = listener_params.n_mfcc;
    int num_filters = listener_params.n_filt;

    // follow frequency config in "sonopy" python package,
    // which use 0 as low & sample_rate as high
    int low_freq = 0;
    int high_freq = sample_rate;

    bool use_preprocess = false;
    bool use_delta = listener_params.use_delta;
    bool use_delta2 = false;
    // get mfcc feature vectors
    mfcc::mfcc<float>(feature_vectors, audio_buffer, sample_rate,
                length_frame, stride, length_FFT, num_coeffs, num_filters,
                low_freq, high_freq,
                use_preprocess, use_delta, use_delta2);

    // append 0 feature vector if not enough
    if (feature_vectors.size() < listener_params.n_features()) {
        int append_num = listener_params.n_features() - feature_vectors.size();
        int feature_size = listener_params.feature_size();

        for (int i = 0; i < append_num; i++) {
            feature_vectors.emplace_back(std::vector<float>(feature_size, 0));
        }
    }

    // check feature vectors shape to align with model input
    assert(feature_vectors.size() == listener_params.n_features());

    for (int i = 0; i < feature_vectors.size(); i++) {
        assert(feature_vectors[i].size() == listener_params.feature_size());
    }

    return;
}


// fulfill feature vectors data to model input tensor
void fill_data(float* out, const std::vector<std::vector<float>> &feature_vectors,
               Settings* s)
{
    int k = 0;
    for (int i = 0; i < feature_vectors.size(); i++) {
        for (int j = 0; j < feature_vectors[i].size(); j++) {
            out[k] = (float)feature_vectors[i][j];
            k++;
        }
    }

    return;
}


//descend order sort for class prediction records
bool compare_conf(std::pair<uint8_t, float> lpred, std::pair<uint8_t, float> rpred)
{
    if (lpred.second < rpred.second)
        return false;
    else
        return true;
}


// Speech Commands Classifier postprocess
void speech_commands_postprocess(const Tensor* score_tensor, std::vector<std::pair<uint8_t, float>> &class_results)
{
    // 1. do following transform to get sorted class index & score:
    //
    //    class = np.argsort(pred, axis=-1)
    //    class = class[::-1]
    //
    const float* data = score_tensor->host<float>();
    auto unit = sizeof(float);
    auto dimType = score_tensor->getDimensionType();

    auto batch   = score_tensor->batch();
    auto channel = score_tensor->channel();
    auto height  = score_tensor->height();
    auto width   = score_tensor->width();

    // batch size should be always 1
    assert(batch == 1);

    int class_size;
    int bytesPerRow, bytesPerImage, bytesPerBatch;
    if (dimType == Tensor::TENSORFLOW) {
        // Tensorflow format tensor, NHWC
        // output is on height dim, so width & channel should be 0
        class_size = height;
        MNN_ASSERT(width == 0);
        MNN_ASSERT(channel == 0);

        //bytesPerRow   = channel * unit;
        //bytesPerImage = width * bytesPerRow;
        //bytesPerBatch = height * bytesPerImage;
        bytesPerBatch   = height * unit;

    } else if (dimType == Tensor::CAFFE) {
        // Caffe format tensor, NCHW
        // output is on channel dim, so width & height should be 0
        class_size = channel;
        MNN_ASSERT(width == 0);
        MNN_ASSERT(height == 0);

        //bytesPerRow   = width * unit;
        //bytesPerImage = height * bytesPerRow;
        //bytesPerBatch = channel * bytesPerImage;
        bytesPerBatch   = channel * unit;

    } else if (dimType == Tensor::CAFFE_C4) {
        MNN_PRINT("Caffe format: NC4HW4, not supported\n");
        exit(-1);
    } else {
        MNN_PRINT("Invalid tensor dim type: %d\n", dimType);
        exit(-1);
    }

    for (int b = 0; b < batch; b++) {
        const float* bytes = data + b * bytesPerBatch / unit;
        //MNN_PRINT("batch %d:\n", b);

        // Get sorted class index & score,
        // just as Python postprocess:
        //
        // class = np.argsort(pred, axis=-1)
        // class = class[::-1]
        //
        uint8_t class_index = 0;
        float max_score = 0.0;
        for (int i = 0; i < class_size; i++) {
            class_results.emplace_back(std::make_pair(i, bytes[i]));
            if (bytes[i] > max_score) {
                class_index = i;
                max_score = bytes[i];
            }
        }
        // descend sort the class prediction list
        std::sort(class_results.begin(), class_results.end(), compare_conf);
    }
    return;
}

#endif  // SPEECH_COMMANDS_H_
