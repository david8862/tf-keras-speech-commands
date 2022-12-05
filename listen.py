#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run speech commands model inference on streaming audio from microphone
"""
import os, argparse, time
import numpy as np
import math
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
        "sensitivity": 0.5,
        "trigger_level": 3,
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

        # load & update audio params
        if self.params_path:
            self.pr = inject_params(self.params_path)
        else:
            self.pr = pr

        # load class names
        self.class_names = get_classes(self.classes_path)
        assert self.class_names[0] == 'background', '1st class should be background.'

        # get listener inference model
        self.model, self.model_format = load_inference_model(self.model_path)
        if self.model_path.endswith('.mnn'):
            #MNN inference engine need create session
            self.session = self.model.createSession()

        # get ThresholdDecoder object for postprocess
        self.threshold_decoder = ThresholdDecoder(self.pr.threshold_config, pr.threshold_center)

        # get TriggerDetector object for postprocess
        self.detector = TriggerDetector(self.chunk_size, self.class_names, self.sensitivity, self.trigger_level)

        # create PyAudio stream
        self.pa = pyaudio.PyAudio()
        self.stream = self.pa.open(rate=16000,
                                   channels=1,
                                   format=pyaudio.paInt16,
                                   input=True,
                                   frames_per_buffer=self.chunk_size)

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
        # and ignore label display
        if class_name == 'background':
            score = 1.0 - score
            class_name = ''

        units = int(round(score * width))
        bar = 'X' * units + '-' * (width - units)
        cutoff = round((1.0 - self.sensitivity) * width)
        print(bar[:cutoff] + bar[cutoff:].replace('X', 'x') + class_name)


    def on_activation(self, index):
        print('command {} detected!'.format(self.class_names[index]))

        activate_audio = 'assets/activate.wav'
        activate_audio = os.path.join(os.path.dirname(os.path.abspath(__file__)), activate_audio)
        self.play_activate_audio(activate_audio)

    def play_activate_audio(self, filename):
        import wave
        CHUNK_SIZE = 1024

        wf = wave.open(filename, 'rb')
        # open stream
        p = pyaudio.PyAudio()
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)
        # read data
        data = wf.readframes(CHUNK_SIZE)

        # play stream
        datas = []
        while len(data) > 0:
            data = wf.readframes(CHUNK_SIZE)
            datas.append(data)

        for d in datas:
            stream.write(d)

        # stop stream
        stream.stop_stream()
        stream.close()

        # close PyAudio
        p.terminate()


    def run(self):
        while True:
            # read audio chunk data from PyAudio stream
            chunk = self.stream.read(self.chunk_size)
            if len(chunk) == 0:
                raise EOFError

            # update mfcc feature with new audio data
            mfccs = self.update_vectors(chunk)
            features = np.expand_dims(mfccs, axis=0).astype(np.float32)

            # run inference to get raw prediction
            output = self.predict(features)
            index = np.argmax(output, axis=-1)
            score = np.max(output, axis=-1)

            # decode non-bg raw score with ThresholdDecoder
            if self.class_names[index] != 'background':
                score = self.threshold_decoder.decode(score)

            # show confidence bar & class label
            self.on_prediction(index, score)

            # update command trigger detector and trigger activation
            if self.detector.update(index, score):
                self.on_activation(index)


class ThresholdDecoder:
    """
    Decode raw network output into a relatively linear threshold using
    This works by estimating the logit normal distribution of network
    activations using a series of averages and standard deviations to
    calculate a cumulative probability distribution

    Background:
    We could simply take the output of the neural network as the confidence of a given
    prediction, but this typically jumps quickly between 0.01 and 0.99 even in cases where
    the network is less confident about a prediction. This is a symptom of the sigmoid squashing
    high values to values close to 1. This ThresholdDecoder measures the average output of
    the network over a dataset and uses that to create a smooth distribution so that an output
    of 80% means that the network output is greater than roughly 80% of the dataset
    """
    def __init__(self, mu_stds, center=0.5, resolution=200, min_z=-4, max_z=4):
        self.min_out = int(min(mu + min_z * std for mu, std in mu_stds))
        self.max_out = int(max(mu + max_z * std for mu, std in mu_stds))
        self.out_range = self.max_out - self.min_out
        self.cd = np.cumsum(self._calc_pd(mu_stds, resolution))
        self.center = center

    def sigmoid(self, x):
        """
        Sigmoid squashing function for scalars
        """
        return 1 / (1 + math.exp(-x))

    def asigmoid(self, x):
        """
        Inverse sigmoid (logit) for scalars
        """
        # check input value to avoid overflow
        return -math.log(1 / x - 1) if (x > 0 and x < 1) else -10

    def pdf(self, x, mu, std):
        """
        Probability density function (normal distribution)
        """
        if std == 0:
            return 0
        return (1.0 / (std * math.sqrt(2 * math.pi))) * np.exp(-(x - mu) ** 2 / (2 * std ** 2))


    def decode(self, raw_output: float) -> float:
        if raw_output == 1.0 or raw_output == 0.0:
            return raw_output
        if self.out_range == 0:
            cp = int(raw_output > self.min_out)
        else:
            ratio = (self.asigmoid(raw_output) - self.min_out) / self.out_range
            ratio = min(max(ratio, 0.0), 1.0)
            cp = self.cd[int(ratio * (len(self.cd) - 1) + 0.5)]
        if cp < self.center:
            return 0.5 * cp / self.center
        else:
            return 0.5 + 0.5 * (cp - self.center) / (1 - self.center)

    def encode(self, threshold: float) -> float:
        threshold = 0.5 * threshold / self.center
        if threshold < 0.5:
            cp = threshold * self.center * 2
        else:
            cp = (threshold - 0.5) * 2 * (1 - self.center) + self.center
        ratio = np.searchsorted(self.cd, cp) / len(self.cd)
        return self.sigmoid(self.min_out + self.out_range * ratio)

    def _calc_pd(self, mu_stds, resolution):
        points = np.linspace(self.min_out, self.max_out, resolution * self.out_range)
        return np.sum([self.pdf(points, mu, std) for mu, std in mu_stds], axis=0) / (resolution * len(mu_stds))



class TriggerDetector(object):
    """
    Reads predictions and detects activations
    This prevents multiple close activations from occurring
    """
    def __init__(self, chunk_size, class_names, sensitivity=0.5, trigger_level=3):
        self.chunk_size = chunk_size
        self.class_names = class_names
        self.sensitivity = sensitivity
        self.trigger_level = trigger_level
        self.activation = 0
        self.record_index = None

    def update(self, index, score):
        """
        Returns whether the new prediction caused an activation
        """
        chunk_activated = score > 1.0 - self.sensitivity

        if (self.class_names[index] != 'background' and index == self.record_index and chunk_activated):
            self.activation += 1
            has_activated = self.activation > self.trigger_level
            if has_activated:
                # reset activation
                self.activation = -(8 * 2048) // self.chunk_size
                return True

        elif self.activation < 0:
            self.activation += 1
        elif self.activation > 0:
            self.activation -= 1

        # record class index for checking
        self.record_index = index
        return False


def main():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description='demo speech commands model (h5/pb/onnx/tflite/mnn) inference on streaming audio from microphone')
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

    parser.add_argument(
        '--sensitivity', type=float, required=False, default=0.5,
        help='model output required to be considered activated. default=%(default)s')

    parser.add_argument(
        '--trigger_level', type=int, required=False, default=3,
        help='number of activated chunks to cause an activation. default=%(default)s')

    args = parser.parse_args()


    # get wrapped listener object
    listener = Listener(**vars(args))

    # run listener loop
    listener.run()



if __name__ == '__main__':
    main()
