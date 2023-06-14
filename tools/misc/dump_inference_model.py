#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dump training checkpoint to inference model
"""
import os, sys, argparse
import tensorflow.keras.backend as K

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..'))
from classifier.model import get_model
from classifier.params import inject_params
from common.utils import get_classes, optimize_tf_gpu

import tensorflow as tf
optimize_tf_gpu(tf, K)


def dump_inference_model(model_type, weights_path, classes_path, params_path, batch_size, output_file):
    class_names = get_classes(classes_path)
    assert class_names[0] == 'background', '1st class should be background.'
    num_classes = len(class_names)

    # load & update audio params
    if params_path:
        inject_params(params_path)

    # load model from training checkpoint
    model = get_model(model_type, num_classes, batch_size=batch_size, weights_path=weights_path)
    model.summary()
    K.set_learning_phase(0)

    # save inference model
    model.save(output_file)


def main():
    parser = argparse.ArgumentParser(description='Dump training checkpoint to inference model, for further convert')
    parser.add_argument('--model_type', type=str, required=False, default='simple_cnn',
        help='wake words model type: simple_cnn/simple_cnn_lite/simple_gru/simple_lstm, default=%(default)s')
    parser.add_argument('--weights_path', type=str, required=True,
        help='training checkpoint model/weights file for dump')
    parser.add_argument('--classes_path', type=str, required=True,
        help='path to class definitions')
    parser.add_argument('--params_path', type=str, required=False, default=None,
        help='path to params json file')
    parser.add_argument('--batch_size', type=int, required=False, default=1,
        help='batch size for inference model, None means dynamic batch. default=%(default)s')
    parser.add_argument('--output_file', type=str, required=True,
        help='output inference model file')

    args = parser.parse_args()

    dump_inference_model(args.model_type, args.weights_path, args.classes_path, args.params_path, args.batch_size, args.output_file)


if __name__ == "__main__":
    main()
