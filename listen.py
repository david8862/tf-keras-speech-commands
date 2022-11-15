#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, argparse, time
import numpy as np
import pyaudio
from tqdm import tqdm
from shutil import get_terminal_size

from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.lite.python import interpreter as interpreter_wrapper
import MNN
import onnxruntime

from classifier.params import pr, inject_params
from common.utils import get_classes, optimize_tf_gpu
from common.model_utils import load_inference_model
from common.data_utils import buffer_to_audio, vectorize_raw, add_deltas

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

optimize_tf_gpu(tf, K)


default_config = {
        "model_path": '',
        "classes_path": os.path.join('configs', 'direction_classes.txt'),
        "params_path": None,
        "chunk_size": 1024,
    }


class Listener(object):
    _defaults = default_config

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        super(Listener, self).__init__()
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides

        # get listener inference model
        self.model, self.model_format = load_inference_model(self.model_path)
        if self.model_path.endswith('.mnn'):
            #MNN inference engine need create session
            self.session = self.model.createSession()

        # create PyAudio stream
        self.pa = pyaudio.PyAudio()
        self.stream = self.pa.open(rate=16000,
                                   channels=1,
                                   format=pyaudio.paInt16,
                                   input=True,
                                   frames_per_buffer=self.chunk_size)

        self.class_names = get_classes(self.classes_path)
        assert self.class_names[0] in ['background', 'others'], '1st class should be background.'

        # load & update audio params
        if self.params_path:
            self.pr = inject_params(self.params_path)
        else:
            self.pr = pr

        # init audio & feature buffer
        self.window_audio = np.array([])
        self.mfccs = np.zeros((self.pr.n_features, self.pr.n_mfcc))

    def update_vectors(self, chunk):
        buffer_audio = buffer_to_audio(chunk)
        self.window_audio = np.concatenate((self.window_audio, buffer_audio))
        #self.window_audio = np.concatenate((self.window_audio[len(buffer_audio):], buffer_audio))

        if len(self.window_audio) >= self.pr.window_samples:
            new_features = vectorize_raw(self.window_audio)
            self.window_audio = self.window_audio[len(new_features) * self.pr.hop_samples:]
            if len(new_features) > len(self.mfccs):
                new_features = new_features[-len(self.mfccs):]
            self.mfccs = np.concatenate((self.mfccs[len(new_features):], new_features))

            if self.pr.use_delta:
                self.mfccs = add_deltas(self.mfccs)

        return np.expand_dims(self.mfccs, axis=-1)

    def predict(self, data):
        # normal keras h5 model
        if self.model_format == 'H5':
            output = self.predict_keras(self.model, data)
        # support of TF 1.x frozen pb model
        elif self.model_format == 'PB':
            output = self.predict_pb(self.model, data)
        # support of tflite model
        elif self.model_format == 'TFLITE':
            output = self.predict_tflite(self.model, data)
        # support of ONNX model
        elif self.model_format == 'ONNX':
            output = self.predict_onnx(self.model, data)
        # support of MNN model
        elif self.model_format == 'MNN':
            output = self.predict_mnn(self.model, self.session, data)
        else:
            raise ValueError('invalid model format')

        return output


    def predict_keras(self, model, data):
        output = model.predict([data])
        return np.squeeze(output)


    def predict_pb(self, model, data):
        # check tf version to be compatible with TF 2.x
        global tf
        if tf.__version__.startswith('2'):
            import tensorflow.compat.v1 as tf
            tf.disable_eager_execution()

        # NOTE: TF 1.x frozen pb graph need to specify input/output tensor name
        # so we need to hardcode the input/output tensor names here to get them from model
        output_tensor_name = 'graph/score_predict/Softmax:0'

        # assume only 1 input tensor for feature vector
        input_tensor_name = 'graph/feature_input:0'

        # get input/output tensors
        feature_input = model.get_tensor_by_name(input_tensor_name)
        output_tensor = model.get_tensor_by_name(output_tensor_name)

        if len(feature_input.shape) == 3:
            # squeeze feature vector to align with RNN input dim
            data = np.squeeze(data, axis=-1)

        with tf.Session(graph=model) as sess:
            output = sess.run(output_tensor, feed_dict={
                feature_input: data
            })

        return np.squeeze(output)


    def predict_tflite(self, interpreter, data):
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        if len(input_details[0]['shape']) == 3:
            # squeeze feature vector to align with RNN input dim
            data = np.squeeze(data, axis=-1)

        interpreter.set_tensor(input_details[0]['index'], data)
        interpreter.invoke()

        output = []
        for output_detail in output_details:
            output_data = interpreter.get_tensor(output_detail['index'])
            output.append(output_data)

        return np.squeeze(output)


    def predict_onnx(self, model, data):
        input_tensors = []
        for i, input_tensor in enumerate(model.get_inputs()):
            input_tensors.append(input_tensor)
        # assume only 1 input tensor for feature vector
        assert len(input_tensors) == 1, 'invalid input tensor number.'

        if len(input_tensors[0].shape) == 3:
            # squeeze feature vector to align with RNN input dim
            data = np.squeeze(data, axis=-1)
        elif len(input_tensors[0].shape) == 4 and input_tensors[0].shape[1] == 1:
            # transpose feature data for NCHW layout
            data = data.transpose((0,3,1,2))

        feed = {input_tensors[0].name: data}
        output = model.run(None, feed)

        return np.squeeze(output)


    def predict_mnn(self, interpreter, session, data):
        from functools import reduce
        from operator import mul

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
                #print("CNN with NCHW input layout")
                batch, channel, feature_num, feature_size = input_shape  #NCHW
            else:
                #print("CNN with NHWC input layout")
                batch, feature_num, feature_size, channel = input_shape  #NHWC
            tmp_input_shape = (batch, feature_num, feature_size, channel)
        else:
            raise ValueError('invalid input tensor shape')

        # create a temp tensor to copy data,
        # use TF NHWC layout to align with image data array
        # TODO: currently MNN python binding have mem leak when creating MNN.Tensor
        # from numpy array, only from tuple is good. So we convert input image to tuple
        input_elementsize = reduce(mul, tmp_input_shape)
        tmp_input = MNN.Tensor(tmp_input_shape, input_tensor.getDataType(),\
                        tuple(data.reshape(input_elementsize, -1)), MNN.Tensor_DimensionType_Tensorflow)

        input_tensor.copyFrom(tmp_input)
        interpreter.runSession(session)

        output = []
        # we only handle single output model
        output_tensor = interpreter.getSessionOutput(session)
        output_shape = output_tensor.getShape()

        assert output_tensor.getDataType() == MNN.Halide_Type_Float

        # copy output tensor to host, for further postprocess
        output_elementsize = reduce(mul, output_shape)
        tmp_output = MNN.Tensor(output_shape, output_tensor.getDataType(),\
                    tuple(np.zeros(output_shape, dtype=float).reshape(output_elementsize, -1)), output_tensor.getDimensionType())

        output_tensor.copyToHostTensor(tmp_output)
        #tmp_output.printTensorData()

        output_data = np.array(tmp_output.getData(), dtype=float).reshape(output_shape)
        output.append(output_data)

        return np.squeeze(output)


    def on_prediction(self, index, score):
        max_width = 80
        width = min(get_terminal_size()[0], max_width)

        class_name = self.class_names[index]
        # for background prediction, just show inversed score
        if class_name == 'background':
            score = 1.0 - score

        units = int(round(score * width))
        bar = 'X' * units + '-' * (width - units)
        print(bar + class_name)
        #cutoff = round((1.0 - self.args.sensitivity) * width)
        #print(bar[:cutoff] + bar[cutoff:].replace('X', 'x'))


    def run(self):
        while True:
            # read audio chunk data from PyAudio stream
            chunk = self.stream.read(self.chunk_size)
            if len(chunk) == 0:
                raise EOFError

            # update mfcc feature with new audio data
            mfccs = self.update_vectors(chunk)
            features = np.expand_dims(mfccs, axis=0).astype(np.float32)

            # run inference and update mfcc feature with new audio data
            output = self.predict(features)

            index = np.argmax(output, axis=-1)
            score = np.max(output, axis=-1)

            self.on_prediction(index, score)



def main():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description='evaluate speech commands classifier model (h5/pb/onnx/tflite/mnn) with test dataset')
    '''
    Command line options
    '''
    parser.add_argument(
        '--model_path', type=str, required=True,
        help='path to model file')

    parser.add_argument(
        '--classes_path', type=str, required=True,
        help='path to class definitions')

    parser.add_argument(
        '--params_path', type=str, required=False, default=None,
        help='path to params json file')

    parser.add_argument(
        '--chunk_size', type=int, required=False, default=1024,
        help='audio samples between inference. default=%(default)s')

    args = parser.parse_args()


    # get wrapped listener object
    listener = Listener(**vars(args))

    listener.run()



if __name__ == '__main__':
    main()
