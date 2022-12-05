#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, argparse
import time
import glob
import numpy as np
from operator import mul
from functools import reduce
import MNN
import onnxruntime
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from tensorflow.lite.python import interpreter as interpreter_wrapper
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
from common.utils import get_classes
from common.model_utils import load_inference_model
from common.data_utils import get_mfcc_feature
from classifier.params import inject_params


def validate_speech_commands_model(model, audio_file, class_names, top_k, loop_count, output_path):
    # prepare input feature vector
    feature_data = get_mfcc_feature(audio_file)
    feature_data = np.expand_dims(feature_data, axis=0)

    # predict once first to bypass the model building time
    model.predict([feature_data])

    # get predict output
    start = time.time()
    for i in range(loop_count):
        prediction = model.predict([feature_data])
    end = time.time()
    print("Average Inference time: {:.8f}ms".format((end - start) * 1000 /loop_count))

    handle_prediction(prediction[0], audio_file, class_names, top_k, output_path)
    return


def validate_speech_commands_model_onnx(model, audio_file, class_names, top_k, loop_count, output_path):
    input_tensors = []
    for i, input_tensor in enumerate(model.get_inputs()):
        input_tensors.append(input_tensor)
    # assume only 1 input tensor for feature vector
    assert len(input_tensors) == 1, 'invalid input tensor number.'

    if len(input_tensors[0].shape) == 3:
        # RNN model has 3 input dim
        batch, feature_num, feature_size = input_tensors[0].shape
    elif len(input_tensors[0].shape) == 4:
        # CNN model has 4 input dim, need to check if layout is NHWC or NCHW
        if input_tensors[0].shape[1] == 1:
            print("CNN with NCHW input layout")
            batch, channel, feature_num, feature_size = input_tensors[0].shape  #NCHW
        else:
            print("CNN with NHWC input layout")
            batch, feature_num, feature_size, channel = input_tensors[0].shape  #NHWC
    else:
        raise ValueError('invalid input tensor shape')

    output_tensors = []
    for i, output_tensor in enumerate(model.get_outputs()):
        output_tensors.append(output_tensor)
    # assume only 1 output tensor
    assert len(output_tensors) == 1, 'invalid output tensor number.'

    # check if classes number match with model prediction
    num_classes = output_tensors[0].shape[-1]
    assert num_classes == len(class_names), 'classes number mismatch with model.'

    # prepare input feature vector
    feature_data = get_mfcc_feature(audio_file)
    feature_data = np.expand_dims(feature_data, axis=0).astype(np.float32)

    if len(input_tensors[0].shape) == 3:
        # squeeze feature vector to align with RNN input dim
        feature_data = np.squeeze(feature_data, axis=-1)
    elif len(input_tensors[0].shape) == 4 and input_tensors[0].shape[1] == 1:
        # transpose feature data for NCHW layout
        feature_data = feature_data.transpose((0,3,1,2))

    feed = {input_tensors[0].name: feature_data}

    # predict once first to bypass the model building time
    prediction = model.run(None, feed)

    start = time.time()
    for i in range(loop_count):
        prediction = model.run(None, feed)

    end = time.time()
    print("Average Inference time: {:.8f}ms".format((end - start) * 1000 /loop_count))

    handle_prediction(prediction[0], audio_file, class_names, top_k, output_path)
    return


def validate_speech_commands_model_pb(model, audio_file, class_names, top_k, loop_count, output_path):
    # check tf version to be compatible with TF 2.x
    global tf
    if tf.__version__.startswith('2'):
        import tensorflow.compat.v1 as tf
        tf.disable_eager_execution()

    # NOTE: TF 1.x frozen pb graph need to specify input/output tensor name
    # so we hardcode the input/output tensor names here to get them from model
    input_tensor_name = 'graph/feature_input:0'
    output_tensor_name = 'graph/score_predict/Softmax:0'

    # We can list operations, op.values() gives you a list of tensors it produces
    # op.name gives you the name. These op also include input & output node
    # print output like:
    # prefix/Placeholder/inputs_placeholder
    # ...
    # prefix/Accuracy/predictions
    #
    # NOTE: prefix/Placeholder/inputs_placeholder is only op's name.
    # tensor name should be like prefix/Placeholder/inputs_placeholder:0

    #for op in model.get_operations():
        #print(op.name, op.values())

    feature_input = model.get_tensor_by_name(input_tensor_name)
    output_tensor = model.get_tensor_by_name(output_tensor_name)

    if len(feature_input.shape) == 3:
        # RNN model has 3 input dim
        batch, feature_num, feature_size = feature_input.shape
    elif len(feature_input.shape) == 4:
        batch, feature_num, feature_size, channel = feature_input.shape  #NHWC
    else:
        raise ValueError('invalid input tensor shape')

    # check if classes number match with model prediction
    num_classes = output_tensor.shape[-1]
    assert num_classes == len(class_names), 'classes number mismatch with model.'

    # prepare input feature vector
    feature_data = get_mfcc_feature(audio_file)
    feature_data = np.expand_dims(feature_data, axis=0)

    if len(feature_input.shape) == 3:
        # squeeze feature vector to align with RNN input dim
        feature_data = np.squeeze(feature_data, axis=-1)

    # predict once first to bypass the model building time
    with tf.Session(graph=model) as sess:
        prediction = sess.run(output_tensor, feed_dict={
            feature_input: feature_data
        })

    start = time.time()
    for i in range(loop_count):
            with tf.Session(graph=model) as sess:
                prediction = sess.run(output_tensor, feed_dict={
                    feature_input: feature_data
                })
    end = time.time()
    print("Average Inference time: {:.8f}ms".format((end - start) * 1000 /loop_count))

    handle_prediction(prediction[0], audio_file, class_names, top_k, output_path)
    return


def validate_speech_commands_model_tflite(interpreter, audio_file, class_names, top_k, loop_count, output_path):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    #print(input_details)
    #print(output_details)

    # check the type of the input tensor
    if input_details[0]['dtype'] == np.float32:
        floating_model = True

    if len(input_details[0]['shape']) == 3:
        # RNN model has 3 input dim
        batch, feature_num, feature_size = input_details[0]['shape']
    elif len(input_details[0]['shape']) == 4:
        batch, feature_num, feature_size, channel = input_details[0]['shape'] #NHWC
    else:
        raise ValueError('invalid input tensor shape')

    # check if classes number match with model prediction
    num_classes = output_details[0]['shape'][-1]
    assert num_classes == len(class_names), 'classes number mismatch with model.'

    # prepare input feature vector
    feature_data = get_mfcc_feature(audio_file)
    feature_data = np.expand_dims(feature_data, axis=0).astype(np.float32)

    if len(input_details[0]['shape']) == 3:
        # squeeze feature vector to align with RNN input dim
        feature_data = np.squeeze(feature_data, axis=-1)

    # predict once first to bypass the model building time
    interpreter.set_tensor(input_details[0]['index'], feature_data)
    interpreter.invoke()

    start = time.time()
    for i in range(loop_count):
        interpreter.set_tensor(input_details[0]['index'], feature_data)
        interpreter.invoke()
    end = time.time()
    print("Average Inference time: {:.8f}ms".format((end - start) * 1000 /loop_count))

    prediction = []
    for output_detail in output_details:
        output_data = interpreter.get_tensor(output_detail['index'])
        prediction.append(output_data)

    handle_prediction(prediction[0], audio_file, class_names, top_k, output_path)
    return


def validate_speech_commands_model_mnn(interpreter, session, audio_file, class_names, top_k, loop_count, output_path):
    # assume only 1 input tensor for feature vector
    input_tensor = interpreter.getSessionInput(session)

    # get & resize input shape
    input_shape = list(input_tensor.getShape())
    if input_shape[0] == 0:
        input_shape[0] = 1
        interpreter.resizeTensor(input_tensor, tuple(input_shape))
        interpreter.resizeSession(session)

    if len(input_shape) == 3:
        # RNN model has 3 input dim
        batch, feature_num, feature_size = input_shape
        tmp_input_shape = (batch, feature_num, feature_size)
    elif len(input_shape) == 4:
        # CNN model has 4 input dim, need to check if layout is NHWC or NCHW
        if input_shape[1] == 1:
            print("CNN with NCHW input layout")
            batch, channel, feature_num, feature_size = input_shape  #NCHW
        else:
            print("CNN with NHWC input layout")
            batch, feature_num, feature_size, channel = input_shape  #NHWC
        tmp_input_shape = (batch, feature_num, feature_size, channel)
    else:
        raise ValueError('invalid input tensor shape')

    # prepare input feature vector
    feature_data = get_mfcc_feature(audio_file)
    feature_data = np.expand_dims(feature_data, axis=0)

    # create a temp tensor to copy data
    # use TF NHWC layout to align with image data array
    # TODO: currently MNN python binding have mem leak when creating MNN.Tensor
    # from numpy array, only from tuple is good. So we convert input image to tuple
    input_elementsize = reduce(mul, tmp_input_shape)
    tmp_input = MNN.Tensor(tmp_input_shape, input_tensor.getDataType(),\
                    tuple(feature_data.reshape(input_elementsize, -1)), MNN.Tensor_DimensionType_Tensorflow)

    # predict once first to bypass the model building time
    input_tensor.copyFrom(tmp_input)
    interpreter.runSession(session)

    start = time.time()
    for i in range(loop_count):
        input_tensor.copyFrom(tmp_input)
        interpreter.runSession(session)
    end = time.time()
    print("Average Inference time: {:.8f}ms".format((end - start) * 1000 /loop_count))

    prediction = []
    output_tensor = interpreter.getSessionOutput(session)
    output_shape = output_tensor.getShape()
    output_elementsize = reduce(mul, output_shape)

    # check if classes number match with model prediction
    num_classes = output_shape[-1]
    assert num_classes == len(class_names), 'classes number mismatch with model.'

    assert output_tensor.getDataType() == MNN.Halide_Type_Float

    # copy output tensor to host, for further postprocess
    tmp_output = MNN.Tensor(output_shape, output_tensor.getDataType(),\
                #np.zeros(output_shape, dtype=float), output_tensor.getDimensionType())
                tuple(np.zeros(output_shape, dtype=float).reshape(output_elementsize, -1)), output_tensor.getDimensionType())

    output_tensor.copyToHostTensor(tmp_output)
    output_data = np.array(tmp_output.getData(), dtype=float).reshape(output_shape)

    prediction.append(output_data)
    handle_prediction(prediction[0], audio_file, class_names, top_k, output_path)
    return



def handle_prediction(prediction, audio_file, class_names, top_k, output_path):
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.splitext(os.path.basename(audio_file))[0]
        output_file = os.path.join(output_path, output_file+'.txt')
        output_file_p = open(output_file, 'w')
    else:
        output_file_p = None

    # get top_k result
    prediction = np.squeeze(prediction)
    sorted_index = np.argsort(prediction)[::-1]
    for i in range(top_k):
        index = sorted_index[i]
        human_string = class_names[index]
        score = prediction[index]
        result_string = '%s: %.3f' % (human_string, score)
        print(result_string)
        if output_file_p:
            output_file_p.write(result_string+'\n')

    if output_file_p:
        output_file_p.close()

    return



def main():
    parser = argparse.ArgumentParser(description='validate speech commands classifier model (h5/pb/onnx/tflite/mnn) with audio file')
    parser.add_argument('--model_path', help='model file to predict', type=str, required=True)
    parser.add_argument('--audio_path', help='input audio file or directory', type=str, required=True)
    parser.add_argument('--classes_path', help='path to class name definitions', type=str, required=True)
    parser.add_argument('--params_path', help='path to params json file', type=str, required=False, default=None)
    parser.add_argument('--top_k', help='top k prediction to print, default=%(default)s.', type=int, required=False, default=1)
    parser.add_argument('--loop_count', help='loop inference for certain times', type=int, default=1)
    parser.add_argument('--output_path', help='output path to save predict result, default=%(default)s', type=str, required=False, default=None)

    args = parser.parse_args()

    class_names = get_classes(args.classes_path)
    assert class_names[0] == 'background', '1st class should be background.'

    # load & update audio params
    if args.params_path:
        inject_params(args.params_path)

    model, _ = load_inference_model(args.model_path)
    if args.model_path.endswith('.mnn'):
        #MNN inference engine need create session
        session = model.createSession()

    # get audio file list or single audio
    if os.path.isdir(args.audio_path):
        audio_files = glob.glob(os.path.join(args.audio_path, '*'))
    else:
        audio_files = [args.audio_path]

    # loop the sample list to predict on each audio
    for audio_file in audio_files:
        # support of tflite model
        if args.model_path.endswith('.tflite'):
            validate_speech_commands_model_tflite(model, audio_file, class_names, args.top_k, args.loop_count, args.output_path)
        # support of MNN model
        elif args.model_path.endswith('.mnn'):
            validate_speech_commands_model_mnn(model, session, audio_file, class_names, args.top_k, args.loop_count, args.output_path)
        # support of TF 1.x frozen pb model
        elif args.model_path.endswith('.pb'):
            validate_speech_commands_model_pb(model, audio_file, class_names, args.top_k, args.loop_count, args.output_path)
        # support of ONNX model
        elif args.model_path.endswith('.onnx'):
            validate_speech_commands_model_onnx(model, audio_file, class_names, args.top_k, args.loop_count, args.output_path)
        # normal keras h5 model
        elif args.model_path.endswith('.h5'):
            validate_speech_commands_model(model, audio_file, class_names, args.top_k, args.loop_count, args.output_path)
        else:
            raise ValueError('invalid model file')


if __name__ == '__main__':
    main()
