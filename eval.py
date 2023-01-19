#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate speech commands model with test dataset
"""
import os, argparse, time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix

import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.lite.python import interpreter as interpreter_wrapper
import MNN
import onnxruntime

from classifier.data import get_dataset
from classifier.params import inject_params
from common.model_utils import load_inference_model
from common.utils import get_classes, optimize_tf_gpu

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

optimize_tf_gpu(tf, K)


def predict_keras(model, data):
    output = model.predict([data])
    pred = np.argmax(output, axis=-1)

    return pred


def predict_pb(model, data):
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
    pred = np.argmax(output, axis=-1)

    return pred


def predict_tflite(interpreter, data):
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

    pred = np.argmax(output[0], axis=-1)

    return pred


def predict_onnx(model, data):
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

    pred = np.argmax(output, axis=-1)

    return pred


def predict_mnn(interpreter, session, data):
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
    pred = np.argmax(output[0], axis=-1)

    return pred


def plot_confusion_matrix(cm, classes, accuracy, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm[np.isnan(cm)] = 0
    trained_classes = classes
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=11)
    tick_marks = np.arange(len(classes))
    plt.xticks(np.arange(len(trained_classes)), classes, rotation=90, fontsize=9)
    plt.yticks(tick_marks, classes, fontsize=9)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, np.round(cm[i, j], 2), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black", fontsize=7)
    plt.ylabel('True label', fontsize=9)
    plt.xlabel('Predicted label', fontsize=9)

    plt.title('Accuracy: ' + str(np.round(accuracy*100, 2)))
    output_path = os.path.join('result', 'confusion_matrix.png')
    os.makedirs('result', exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    #plt.show()

    # close the plot
    plt.close()
    return


def evaluate_accuracy(model, model_format, eval_dataset, class_names):
    if model_format == 'MNN':
        #MNN inference engine need create session
        session = model.createSession()

    correct = 0.0
    target_list = []
    pred_list = []

    x_eval, y_eval = eval_dataset
    pbar = tqdm(total=len(x_eval))
    for i in range(len(x_eval)):
        feature = np.expand_dims(x_eval[i], axis=0).astype(np.float32)
        target = y_eval[i]

        # normal keras h5 model
        if model_format == 'H5':
            pred = predict_keras(model, feature)
        # support of TF 1.x frozen pb model
        elif model_format == 'PB':
            pred = predict_pb(model, feature)
        # support of tflite model
        elif model_format == 'TFLITE':
            pred = predict_tflite(model, feature)
        # support of ONNX model
        elif model_format == 'ONNX':
            pred = predict_onnx(model, feature)
        # support of MNN model
        elif model_format == 'MNN':
            pred = predict_mnn(model, session, feature)
        else:
            raise ValueError('invalid model format')

        correct += float(np.equal(pred, target).astype(np.int32).sum())
        target_list.append(target)
        pred_list.append(pred)

        pbar.set_description('Evaluate acc: %06.4f' % (correct/(i + 1)))
        pbar.update(1)
    pbar.close()

    val_acc = correct / len(x_eval)
    print('Test set accuracy: {}/{} ({:.2f}%)'.format(
        correct, len(x_eval), val_acc*100))

    # Plot accuracy & confusion matrix
    confusion_mat = confusion_matrix(y_true=np.squeeze(target_list), y_pred=np.squeeze(pred_list), labels=list(range(len(class_names))))
    plot_confusion_matrix(confusion_mat, class_names, val_acc, normalize=True)

    return val_acc



def main():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS, description='evaluate speech commands classifier model (h5/pb/onnx/tflite/mnn) with test dataset')
    '''
    Command line options
    '''
    parser.add_argument(
        '--model_path', type=str, required=True,
        help='path to model file')

    parser.add_argument(
        '--dataset_path', type=str, required=True,
        help='path to evaluation audio dataset')

    parser.add_argument(
        '--classes_path', type=str, required=True,
        help='path to class definitions')

    parser.add_argument(
        '--params_path', type=str, required=False, default=None,
        help='path to params json file')

    args = parser.parse_args()

    # param parse
    class_names = get_classes(args.classes_path)
    assert class_names[0] == 'background', '1st class should be background.'

    # load & update audio params
    if args.params_path:
        inject_params(args.params_path)

    # get eval model
    model, model_format = load_inference_model(args.model_path)

    # get eval dataset
    x_eval, y_eval, _, _ = get_dataset(args.dataset_path, class_names)

    start = time.time()
    evaluate_accuracy(model, model_format, (x_eval, y_eval), class_names)
    end = time.time()
    print("Evaluation time cost: {:.6f}s".format(end - start))


if __name__ == '__main__':
    main()
