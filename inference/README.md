## C++ on-device (X86/ARM) inference app for Speech Commands recognition modelset

Here are some C++ implementation of the on-device inference for trained speech commands recognition model, including single wav audio file inference and audio stream input inference (from microphone with ALSA). Now we have 2 approaches with different inference engine for that:

* Tensorflow-Lite (verified on tag: v2.6.0)
* [MNN](https://github.com/alibaba/MNN) from Alibaba (verified on release: [1.0.0](https://github.com/alibaba/MNN/releases/tag/1.0.0))

**NOTE**: Currently the TFLite demo app support RNN based (GRU/LSTM) model, and MNN demo app support CNN based model


### Tensorflow-Lite

1. Install ALSA lib and Build TF-Lite lib

We can do either native compile for X86 or cross-compile for ARM

```
### for cross-compile you may need to manually prepare ALSA libs
# apt install libasound2-dev alsa-utils

# git clone https://github.com/tensorflow/tensorflow <Path_to_TF>
# cd <Path_to_TF>
# git checkout v2.6.0
# ./tensorflow/lite/tools/make/download_dependencies.sh
# make -f tensorflow/lite/tools/make/Makefile   #for X86 native compile
# ./tensorflow/lite/tools/make/build_rpi_lib.sh #for ARM cross compile, e.g Rasperberry Pi
```

you can also create your own build script for new ARM platform, like:

```shell
# vim ./tensorflow/lite/tools/make/build_my_arm_lib.sh

#!/bin/bash -x
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../../../.."
make CC_PREFIX=/root/toolchain/aarch64-linux-gnu/bin/aarch64-linux-gnu- -j 3 -f tensorflow/lite/tools/make/Makefile TARGET=myarm TARGET_ARCH=aarch64 $@
```

**NOTE:**
* Using Makefile to build TensorFlow Lite is deprecated since Aug 2021. So v2.6.0 should be the last major version to support Makefile build (cmake is enabled on new version)
* by default TF-Lite build only generate static lib (.a), but we can do minor change in Makefile to generate .so shared lib together, as follow:

```diff
diff --git a/tensorflow/lite/tools/make/Makefile b/tensorflow/lite/tools/make/Makefile
index 662c6bb5129..83219a42845 100644
--- a/tensorflow/lite/tools/make/Makefile
+++ b/tensorflow/lite/tools/make/Makefile
@@ -99,6 +99,7 @@ endif
 # This library is the main target for this makefile. It will contain a minimal
 # runtime that can be linked in to other programs.
 LIB_NAME := libtensorflow-lite.a
+SHARED_LIB_NAME := libtensorflow-lite.so

 # Benchmark static library and binary
 BENCHMARK_LIB_NAME := benchmark-lib.a
@@ -301,6 +302,7 @@ BINDIR := $(GENDIR)bin/
 LIBDIR := $(GENDIR)lib/

 LIB_PATH := $(LIBDIR)$(LIB_NAME)
+SHARED_LIB_PATH := $(LIBDIR)$(SHARED_LIB_NAME)
 BENCHMARK_LIB := $(LIBDIR)$(BENCHMARK_LIB_NAME)
 BENCHMARK_BINARY := $(BINDIR)$(BENCHMARK_BINARY_NAME)
 BENCHMARK_PERF_OPTIONS_BINARY := $(BINDIR)$(BENCHMARK_PERF_OPTIONS_BINARY_NAME)
@@ -344,7 +346,7 @@ $(OBJDIR)%.o: %.c
        $(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

 # The target that's compiled if there's no command-line arguments.
-all: $(LIB_PATH)  $(MINIMAL_BINARY) $(BENCHMARK_BINARY) $(BENCHMARK_PERF_OPTIONS_BINARY)
+all: $(LIB_PATH) $(SHARED_LIB_PATH) $(MINIMAL_BINARY) $(BENCHMARK_BINARY) $(BENCHMARK_PERF_OPTIONS_BINARY)

 # The target that's compiled for micro-controllers
 micro: $(LIB_PATH)
@@ -361,7 +363,14 @@ $(LIB_PATH): tensorflow/lite/experimental/acceleration/configuration/configurati
        @mkdir -p $(dir $@)
        $(AR) $(ARFLAGS) $(LIB_PATH) $(LIB_OBJS)

-lib: $(LIB_PATH)
+$(SHARED_LIB_PATH): tensorflow/lite/schema/schema_generated.h $(LIB_OBJS)
+       @mkdir -p $(dir $@)
+       $(CXX) $(CXXFLAGS) -shared -o $(SHARED_LIB_PATH) $(LIB_OBJS)
+$(SHARED_LIB_PATH): tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h $(LIB_OBJS)
+       @mkdir -p $(dir $@)
+       $(CXX) $(CXXFLAGS) -shared -o $(SHARED_LIB_PATH) $(LIB_OBJS)
+
+lib: $(LIB_PATH) $(SHARED_LIB_PATH)

 $(MINIMAL_BINARY): $(MINIMAL_OBJS) $(LIB_PATH)
        @mkdir -p $(dir $@)
```


2. Build cJSON lib to parse .json params config file

We can do either native compile for X86 or cross-compile for ARM

```
# git clone https://github.com/DaveGamble/cJSON.git
# cd cJSON/
# mkdir build
# cd build/
# cmake [-DCMAKE_TOOLCHAIN_FILE=<cross-compile toolchain file> -DCMAKE_INSTALL_PREFIX=<cJSON path>/build/install/] ..
# make && make install
```

on X86 PC environment (Ubuntu), still need to add `/usr/local/lib` into `/etc/ld.so.conf.d/x86_64-linux-gnu.conf` and run `ldconfig` to make libcjson.so discoverable when execute binary. for cross-compile you need to manually config path of the lib & header file.


3. Build demo inference application

```
# cd tf-keras-speech-commands/inference/tflite
# mkdir build && cd build
# cmake -DTF_ROOT_PATH=<Path_to_TF> [-DCMAKE_TOOLCHAIN_FILE=<cross-compile toolchain file>] [-DTARGET_PLAT=<target>] ..
# make
```
If you want to do cross compile for ARM platform, "CMAKE_TOOLCHAIN_FILE" and "TARGET_PLAT" should be specified. Refer [CMakeLists.txt](https://github.com/david8862/tf-keras-speech-commands/blob/master/inference/tflite/CMakeLists.txt) for details.


4. Convert trained speech commands model to TFLite model

    ```
    # cd tf-keras-speech-commands/tools/model_converter/
    # python custom_tflite_convert.py --keras_model_file=model.h5 --output_file=model.tflite
    ```


5. Run validate script to check TFLite model

```
# cd tf-keras-speech-commands/tools/evaluation/
# python validate_speech_commands.py --model_path=model.tflite --audio_path=../../example/right_1.wav --classes_path=../../configs/direction_classes.txt --params_path=../../configs/params.json --top_k=2 --loop_count=5
```

#### You can also use [eval.py](https://github.com/david8862/tf-keras-speech-commands#evaluation) or [listen.py](https://github.com/david8862/tf-keras-speech-commands/blob/master/listen.py) to do evaluation on the TFLite model


6. Run single wav file inference app, or put assets to ARM board and run if cross-compile
```
# cd tf-keras-speech-commands/inference/tflite/build
# ./speech_commands -h
Usage: speech_commands
--tflite_model, -m: model_name.tflite
--params_file, -p: params.json
--classes, -l: classes labels for the model
--top_k, -k: show top k classes result
--wav_file, -i: test.wav
--allow_fp16, -f: [0|1], allow running fp32 models with fp16 or not
--threads, -t: number of threads
--count, -c: loop interpreter->Invoke() for certain times
--warmup_runs, -w: number of warmup runs
--result, -r: result txt file to save detection output
--verbose, -v: [0|1] print more information

# ./speech_commands -m model.tflite -p ../../../configs/params.json -l ../../../configs/direction_classes.txt -k 2 -i ../../../example/right_1.wav -v 0
Loaded model model.tflite
resolved reporter
params json parsed
num_classes: 5

Input audio info:
|======================================|
Num Channels: 1
Num Samples Per Channel: 16000
Sample Rate: 16000
Bit Depth: 16
Length in Seconds: 1
|======================================|
feature vectors extraction time:3.051 ms
invoked average time:2.216 ms
speech_commands_postprocess time: 0.001 ms
Inferenced class:
right: 0.999427
left: 0.000572826
```

7. Run ALSA audio stream input inference app
```
# cd tf-keras-speech-commands/inference/tflite/build
# ./speech_commands_alsa -h
Usage: speech_commands_alsa
--tflite_model, -m: model_name.tflite
--params_file, -p: params.json
--classes, -l: classes labels for the model
--alsa_device, -d: ALSA device name
--chunk_size, -c: audio samples between inferences
--sensitivity, -s: model output required to be considered activated
--trigger_level, -g: number of activated chunks to cause an activation
--fast_feature, -e: [0|1], use fast feature extraction or not
--threads, -t: number of threads
--allow_fp16, -f: [0|1], allow running fp32 models with fp16 or not
--verbose, -v: [0|1] print more information

# ./speech_commands_alsa -m model.tflite -p ../../../configs/params.json -l ../../../configs/direction_classes.txt -d hw:0,0 -c 1024 -s 0.5 -g 3 -e 0 -v 1
Loaded model model.tflite
resolved reporter
input tensor info: name feature_input, type kTfLiteFloat32, dim_size 3, batch 1, feature_num 30, feature_size 20
output tensor info: name Identity, type kTfLiteFloat32, dim_size 2, batch 1, length 5
params json parsed
num_classes: 5
PCM buffer time: 500000
PCM sample period size: 1024
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
XX------------------------------------------------------------------------------
XXXXXXXXXXXXXXXXXX--------------------------------------------------------------
XXXXXXXXXXXXXXXXXXXXXXX---------------------------------------------------------
XXXXXXXXXXXXXXXXXXXXXXXXXXXXX---------------------------------------------------
XXXXXXXXXXXXXXXX----------------------------------------------------------------right
XXXXXXXXXXXXXXXXX---------------------------------------------------------------right
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX----------------------------------------------right
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXxx--------------------------------------right
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXxxx-------------------------------------right
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXxxx-------------------------------------right
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx---------------------------------------right
command right detected!
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX--------------------------------------------right
XXXXXXXXXXXXXXXXXXXXXXXXXX------------------------------------------------------right
XXXXXXXXXXXXXXXXXXXX------------------------------------------------------------right
XX------------------------------------------------------------------------------
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
```

here `alsa_device` could be checked with ALSA arecord tool:

```
# arecord -l
**** List of CAPTURE Hardware Devices ****
card 3: ArrayUAC10 [ReSpeaker 4 Mic Array (UAC1.0)], device 0: USB Audio [USB Audio]
  Subdevices: 1/1
  Subdevice #0: subdevice #0
```

then `--alsa_device` could be set to `hw:3,0` or `plughw:3,0`.




### MNN

1. Install Python runtime, ALSA lib and Build libMNN

Refer to [MNN build guide](https://www.yuque.com/mnn/cn/build_linux), we need to prepare cmake & protobuf first for MNN build. And since MNN support both X86 & ARM platform, we can do either native compile or ARM cross-compile

```
### for cross-compile you may need to manually prepare ALSA libs
# apt install libasound2-dev alsa-utils
#
# apt install cmake autoconf automake libtool ocl-icd-opencl-dev
# wget https://github.com/google/protobuf/releases/download/v3.4.1/protobuf-cpp-3.4.1.tar.gz
# tar xzvf protobuf-cpp-3.4.1.tar.gz
# cd protobuf-3.4.1
# ./autogen.sh
# ./configure && make && make check && make install && ldconfig
# pip install --upgrade pip && pip install --upgrade mnn

# git clone https://github.com/alibaba/MNN.git <Path_to_MNN>
# cd <Path_to_MNN>
# ./schema/generate.sh
# ./tools/script/get_model.sh  # optional
# mkdir build && cd build
# cmake [-DCMAKE_TOOLCHAIN_FILE=<cross-compile toolchain file>]
        [-DMNN_BUILD_QUANTOOLS=ON -DMNN_BUILD_CONVERTER=ON -DMNN_BUILD_BENCHMARK=ON -DMNN_BUILD_TRAIN=ON -DMNN_BUILD_TRAIN_MINI=ON -DMNN_USE_OPENCV=OFF] ..
        && make -j4

### MNN OpenCL backend build
# apt install ocl-icd-opencl-dev
# cmake [-DCMAKE_TOOLCHAIN_FILE=<cross-compile toolchain file>]
        [-DMNN_OPENCL=ON -DMNN_SEP_BUILD=OFF -DMNN_USE_SYSTEM_LIB=ON] ..
        && make -j4
```
If you want to do cross compile for ARM platform, "CMAKE_TOOLCHAIN_FILE" should be specified

"MNN_BUILD_QUANTOOLS" is for enabling MNN Quantization tool

"MNN_BUILD_CONVERTER" is for enabling MNN model converter

"MNN_BUILD_BENCHMARK" is for enabling on-device inference benchmark tool

"MNN_BUILD_TRAIN" related are for enabling MNN training tools


2. Build cJSON lib to parse .json params config file

We can do either native compile for X86 or cross-compile for ARM

```
# git clone https://github.com/DaveGamble/cJSON.git
# cd cJSON/
# mkdir build
# cd build/
# cmake [-DCMAKE_TOOLCHAIN_FILE=<cross-compile toolchain file> -DCMAKE_INSTALL_PREFIX=<cJSON path>/build/install/] ..
# make && make install
```

on X86 PC environment (Ubuntu), still need to add `/usr/local/lib` into `/etc/ld.so.conf.d/x86_64-linux-gnu.conf` and run `ldconfig` to make libcjson.so discoverable when execute binary. for cross-compile you need to manually config path of the lib & header file.


3. Build demo inference application
```
# cd tf-keras-speech-commands/inference/MNN
# mkdir build && cd build
# cmake -DMNN_ROOT_PATH=<Path_to_MNN> [-DCMAKE_TOOLCHAIN_FILE=<cross-compile toolchain file>] ..
# make
```


4. Convert trained speech commands model to MNN model

Refer to [Tensorflow model convert](https://github.com/david8862/tf-keras-speech-commands#tensorflow-model-convert), [ONNX model convert](https://github.com/david8862/tf-keras-speech-commands#onnx-model-convert) and [MNN model convert](https://www.yuque.com/mnn/cn/model_convert), we can use 2 approach to convert MNN model:

* convert keras .h5 model to tensorflow frozen pb model:

    ```
    # python keras_to_tensorflow.py
        --input_model="path/to/keras/model.h5"
        --output_model="path/to/save/model.pb"
    ```

* convert TF pb model to MNN model:

    ```
    # mnnconvert -f TF --modelFile model.pb --MNNModel model.pb.mnn
    ```
or

* convert keras .h5 model to onnx model:

    ```
    # python keras_to_onnx.py
        --keras_model_file="path/to/keras/model.h5"
        --output_file="path/to/save/model.onnx"
        --op_set=13
        --inputs_as_nchw
    ```

* convert onnx model to MNN model:

    ```
    # mnnconvert -f ONNX --modelFile model.onnx --MNNModel model.pb.mnn
    ```

MNN support Post Training Integer quantization, so we can use its python CLI interface to do quantization on the generated .mnn model to get quantized .mnn model for ARM acceleration . A json config file [quantizeConfig.json](https://github.com/david8862/tf-keras-speech-commands/blob/master/inference/MNN/configs/quantizeConfig.json) is needed to describe the feeding data:

* Quantized MNN model:

    ```
    # cd <Path_to_MNN>/build/
    # ./quantized.out model.pb.mnn model_quant.pb.mnn quantizeConfig.json
    ```
    or

    ```
    # mnnquant model.pb.mnn model_quant.pb.mnn quantizeConfig.json
    ```

5. Run validate script to check MNN model
```
# cd tf-keras-speech-commands/tools/evaluation/
# python validate_speech_commands.py --model_path=model.pb.mnn --audio_path=../../example/right_1.wav --classes_path=../../configs/direction_classes.txt --params_path=../../configs/params.json --top_k=2 --loop_count=5
```


#### You can also use [eval.py](https://github.com/david8862/tf-keras-speech-commands#evaluation) or [listen.py](https://github.com/david8862/tf-keras-speech-commands/blob/master/listen.py) to do evaluation on the MNN model


6. Run single wav file inference app, or put assets to ARM board and run if cross-compile
```
# cd tf-keras-speech-commands/inference/MNN/build
# ./speech_commands -h
Usage: speech_commands
--mnn_model, -m: model_name.mnn
--params_file, -p: params.json
--classes, -l: classes labels for the model
--top_k, -k: show top k classes result
--wav_file, -i: test.wav
--allow_fp16, -f: [0|1], allow running fp32 models with fp16 or not
--threads, -t: number of threads
--count, -c: loop interpreter->Invoke() for certain times
--warmup_runs, -w: number of warmup runs
--result, -r: result txt file to save detection output
--verbose, -v: [0|1] print more information


# ./speech_commands -m model.pb.mnn -p ../../../configs/params.json -l ../../../configs/direction_classes.txt -k 2 -i ../../../example/right_1.wav -v 0
feature_input: name:feature_input, width:20, height:30, channel:1, dim_type:CAFFE
params json parsed
num_classes: 5

Input audio info:
|======================================|
Num Channels: 1
Num Samples Per Channel: 16000
Sample Rate: 16000
Bit Depth: 16
Length in Seconds: 1
|======================================|
feature vectors extraction time: 3.199000 ms
model invoke average time: 0.575000 ms
output tensor: name:score_predict/Softmax, width:1, height:5, channel:1, dim_type:TENSORFLOW
Tensorflow format: NHWC
batch 0:
speech_commands_postprocess time: 0.584000 ms
Inferenced class:
right: 0.999882
background: 0.000089
```
Here the [classes](https://github.com/david8862/tf-keras-speech-commands/blob/master/configs/direction_classes.txt) & [params](https://github.com/david8862/tf-keras-speech-commands/blob/master/configs/params.json) file format are the same as used in training part


7. Run ALSA audio stream input inference app
```
# cd tf-keras-speech-commands/inference/MNN/build
# ./speech_commands_alsa -h
Usage: speech_commands_alsa
--mnn_model, -m: model_name.mnn
--params_file, -p: params.json
--classes, -l: classes labels for the model
--alsa_device, -d: ALSA device name
--chunk_size, -c: audio samples between inferences
--sensitivity, -s: model output required to be considered activated
--trigger_level, -g: number of activated chunks to cause an activation
--fast_feature, -e: [0|1], use fast feature extraction or not
--threads, -t: number of threads
--allow_fp16, -f: [0|1], allow running fp32 models with fp16 or not
--verbose, -v: [0|1] print more information

# ./speech_commands_alsa -m model.pb.mnn -p ../../../configs/params.json -l ../../../configs/direction_classes.txt -d hw:0,0 -c 1024 -s 0.5 -g 3 -e 0 -v 1
feature_input: name:feature_input, width:20, height:30, channel:1, dim_type:CAFFE
params json parsed
num_classes: 5
output tensor: name:score_predict/Softmax, width:1, height:5, channel:1, dim_type:TENSORFLOW
Tensorflow format: NHWC
PCM buffer time: 500000
PCM sample period size: 1024
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
XX------------------------------------------------------------------------------
XXXXXXXXXXXXXXXXXX--------------------------------------------------------------
XXXXXXXXXXXXXXXXXXXXXXX---------------------------------------------------------
XXXXXXXXXXXXXXXXXXXXXXXXXXXXX---------------------------------------------------
XXXXXXXXXXXXXXXX----------------------------------------------------------------right
XXXXXXXXXXXXXXXXX---------------------------------------------------------------right
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX----------------------------------------------right
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXxx--------------------------------------right
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXxxx-------------------------------------right
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXxxx-------------------------------------right
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx---------------------------------------right
command right detected!
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX--------------------------------------------right
XXXXXXXXXXXXXXXXXXXXXXXXXX------------------------------------------------------right
XXXXXXXXXXXXXXXXXXXX------------------------------------------------------------right
XX------------------------------------------------------------------------------
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
```

here `alsa_device` could be checked with ALSA arecord tool:

```
# arecord -l
**** List of CAPTURE Hardware Devices ****
card 3: ArrayUAC10 [ReSpeaker 4 Mic Array (UAC1.0)], device 0: USB Audio [USB Audio]
  Subdevices: 1/1
  Subdevice #0: subdevice #0
```

then `--alsa_device` could be set to `hw:3,0` or `plughw:3,0`.

