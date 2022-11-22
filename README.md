# TF Keras Speech Commands recognition Modelset

## Introduction

An end-to-end speech commands recognition pipeline. Implement with tf.keras, including model training/tuning, model evaluation, streaming audio demo, trained model export (PB/ONNX/TFLITE) and on device deployment (TFLITE/MNN). Support both CNN & RNN model type:

#### Model Type
- [x] Simple CNN
- [x] Simple GRU
- [x] Simple LSTM


## Guide of train/evaluate/demo

### Train

1. Install requirements on Ubuntu 18.04/20.04:

```
# pip install -r requirements.txt
```

2. Prepare dataset and class names file

    * Get .wav format speech command audio sample files (e.g [Google Speech Commands](http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz)) and place at `<dataset path>/sounds/` with 1 folder for a command. The folder name should be using the command class name and there should be a `background` class folder for many non-command audio samples, like:

    ```
    <dataset path>/
    └── sounds
        ├── background
        │   ├── background_1.wav
        │   ├── background_2.wav
        │   ├── background_3.wav
        │   └── ...
        ├── command1
        │   ├── command1_1.wav
        │   ├── command1_2.wav
        │   ├── command1_3.wav
        │   └── ...
        ├── command2
        │   ├── command2_1.wav
        │   ├── command2_2.wav
        │   └── ...
        │
        └──...
    ```

    **NOTE**:
    1. Audio process pipeline parameters for this project (audio format/feature params/postprocess params) are set in [params.py](https://github.com/david8862/tf-keras-speech-commands/blob/master/classifier/params.py) and could be reloaded with a json format config file (refer to [params.json](https://github.com/david8862/tf-keras-speech-commands/blob/master/configs/params.json)). The .wav audio sample format (audio_length/sample_rate/sample_depth) should be aligned with your params.
    2. The `background` class is mandatory for real time inference, and generally background samples should be much more than command samples to cover the real world non-command cases. For example, you can choose 4 direction commands (up/down/left/right) samples in [Google Speech Commands](http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz) as speech commands and put all the other commands & noise samples into the background class. And the train/val/test dataset path should follow the same structure.

    For class names file format, refer to [direction_classes.txt](https://github.com/david8862/tf-keras-speech-commands/blob/master/configs/direction_classes.txt)


3. [train.py](https://github.com/david8862/tf-keras-speech-commands/blob/master/train.py)

```
# python train.py -h
usage: train.py [-h] [--model_type MODEL_TYPE] [--weights_path WEIGHTS_PATH]
                --train_data_path TRAIN_DATA_PATH
                [--val_data_path VAL_DATA_PATH]
                [--val_split VAL_SPLIT]
                --classes_path CLASSES_PATH
                [--params_path PARAMS_PATH]
                [--background_bias BACKGROUND_BIAS]
                [--batch_size BATCH_SIZE]
                [--optimizer {adam,rmsprop,sgd}]
                [--learning_rate LEARNING_RATE]
                [--decay_type {None,cosine,exponential,polynomial,piecewise_constant}]
                [--epochs EPOCHS]

optional arguments:
  -h, --help            show this help message and exit
  --model_type MODEL_TYPE
                        classifier model type: simple_cnn/simple_gru/simple_lstm, default=simple_cnn
  --weights_path WEIGHTS_PATH
                        Pretrained model/weights file for fine tune
  --train_data_path TRAIN_DATA_PATH
                        path to train dataset
  --val_data_path VAL_DATA_PATH
                        path to val dataset
  --val_split VAL_SPLIT
                        validation data persentage in dataset if no val dataset provide, default=0.15
  --classes_path CLASSES_PATH
                        path to class definitions
  --params_path PARAMS_PATH
                        path to params json file
  --background_bias BACKGROUND_BIAS
                        background loss bias (0~1) when training. lower values may cause more false positives if set, default=None
  --batch_size BATCH_SIZE
                        Batch size for train, default=512
  --optimizer {adam,rmsprop,sgd}
                        optimizer for training (adam/rmsprop/sgd), default=adam
  --learning_rate LEARNING_RATE
                        Initial learning rate, default=0.001
  --decay_type {None,cosine,exponential,polynomial,piecewise_constant}
                        Learning rate decay type, default=None
  --epochs EPOCHS       Total training epochs, default=100
```

Following is reference config cmd for training simple_gru model:
```
# python train.py --model_type=simple_gru --train_data_path=train_data/ --val_data_path=val_data/ --classes_path=configs/direction_classes.txt --params_path=configs/params.json --background_bias=0.9
```

Checkpoints during training could be found at `logs/000/`. Choose a best one as result



### Evaluation
Use [eval.py](https://github.com/david8862/tf-keras-speech-commands/blob/master/eval.py) to do evaluation on the trained model with test dataset:

```
# python eval.py -h
usage: eval.py [-h] --model_path MODEL_PATH
               --dataset_path DATASET_PATH
               --classes_path CLASSES_PATH
               [--params_path PARAMS_PATH]
evaluate speech commands classifier model (h5/pb/onnx/tflite/mnn) with test dataset

optional arguments:
  -h, --help            show this help message and exit
  --model_path MODEL_PATH
                        path to model file
  --dataset_path DATASET_PATH
                        path to evaluation audio dataset
  --classes_path CLASSES_PATH
                        path to class definitions
  --params_path PARAMS_PATH
                        path to params json file
```

Reference cmd:

```
# python eval.py --model_path=model.h5 --dataset_path=test_data/ --classes_path=configs/direction_classes.txt --params_path=configs/params.json
```

You can also use [validate_speech_commands.py](https://github.com/david8862/tf-keras-speech-commands/blob/master/tools/evaluation/validate_speech_commands.py) to validate on single wav file or files:

```
# cd tools/evaluation/ && python validate_speech_commands.py -h
usage: validate_speech_commands.py [-h] --model_path MODEL_PATH
                                   --audio_path AUDIO_PATH
                                   --classes_path CLASSES_PATH
                                   [--params_path PARAMS_PATH]
                                   [--top_k TOP_K]
                                   [--loop_count LOOP_COUNT]
                                   [--output_path OUTPUT_PATH]

validate speech commands classifier model (h5/pb/onnx/tflite/mnn) with audio file

optional arguments:
  -h, --help            show this help message and exit
  --model_path MODEL_PATH
                        model file to predict
  --audio_path AUDIO_PATH
                        input audio file or directory
  --classes_path CLASSES_PATH
                        path to class name definitions
  --params_path PARAMS_PATH
                        path to params json file
  --top_k TOP_K         top k prediction to print, default=1.
  --loop_count LOOP_COUNT
                        loop inference for certain times
  --output_path OUTPUT_PATH
                        output path to save predict result, default=None
```


### Demo
Run live demo with trained model on streaming audio from microphone. This would be more effictive to verify model performance in real world:

[listen.py](https://github.com/david8862/tf-keras-speech-commands/blob/master/listen.py)

```
# python listen.py -h
usage: listen.py [-h] --model_path MODEL_PATH --classes_path CLASSES_PATH [--params_path PARAMS_PATH] [--chunk_size CHUNK_SIZE] [--sensitivity SENSITIVITY] [--trigger_level TRIGGER_LEVEL]

demo speech commands model (h5/pb/onnx/tflite/mnn) inference on streaming audio from microphone

optional arguments:
  -h, --help            show this help message and exit
  --model_path MODEL_PATH
                        path to model file
  --classes_path CLASSES_PATH
                        path to class definitions
  --params_path PARAMS_PATH
                        path to params json file
  --chunk_size CHUNK_SIZE
                        audio samples between inference. default=1024
  --sensitivity SENSITIVITY
                        model output required to be considered activated. default=0.5
  --trigger_level TRIGGER_LEVEL
                        number of activated chunks to cause an activation. default=3

# python listen.py --model_path=model.h5 --classes_path=configs/direction_classes.txt --params_path=configs/params.json
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
X-------------------------------------------------------------------------------
XXXXX---------------------------------------------------------------------------
XXXXXXXX------------------------------------------------------------------------
XXXXXXXXXXXXXXXXX---------------------------------------------------------------
XXXXXXXXXXX---------------------------------------------------------------------
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXxxxxxxx---------------------------------right
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXxxxxxxxxxxx-----------------------------right
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXxxxxxxxxxxxxx---------------------------right
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXxxxxxxxxxxxxxx--------------------------right
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXxxxxxxxxxxxxxxx-------------------------right
command right detected!
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXxxxxxxxxxxxxxxxx------------------------right
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXxxxxxxxxxxxxxx--------------------------right
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXxxxxxxxxxxxx----------------------------right
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXxxxxxxx---------------------------------right
XX------------------------------------------------------------------------------
X-------------------------------------------------------------------------------
X-------------------------------------------------------------------------------
--------------------------------------------------------------------------------
```


### Tensorflow model convert
Using [keras_to_tensorflow.py](https://github.com/david8862/tf-keras-speech-commands/blob/master/tools/model_converter/keras_to_tensorflow.py) to convert the tf.keras .h5 model to tensorflow frozen pb model:
```
# python keras_to_tensorflow.py
    --input_model="path/to/keras/model.h5"
    --output_model="path/to/save/model.pb"
```

### ONNX model convert
Using [keras_to_onnx.py](https://github.com/david8862/tf-keras-speech-commands/blob/master/tools/model_converter/keras_to_onnx.py) to convert the tf.keras .h5 model to ONNX model:
```
### need to set environment TF_KERAS=1 for tf.keras model
# export TF_KERAS=1
# python keras_to_onnx.py
    --keras_model_file="path/to/keras/model.h5"
    --output_file="path/to/save/model.onnx"
    --op_set=11
```
by default, the converted ONNX model follows TF NHWC layout. You can also use `--inputs_as_nchw` to convert input layout to NCHW.

You can also use [eval.py](https://github.com/david8862/tf-keras-speech-commands/blob/master/eval.py) to do evaluation on the pb & onnx inference model

