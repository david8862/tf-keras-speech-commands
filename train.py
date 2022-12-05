#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train Speech Commands model for your own dataset.
"""
import os, sys, argparse
import numpy as np
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TerminateOnNaN
import tensorflow.keras.backend as K

from classifier.model import get_model
from classifier.data import get_dataset
from classifier.loss import SparseCategoricalCrossEntropy, WeightedSparseCategoricalCrossEntropy
from classifier.params import inject_params
from common.utils import get_classes, optimize_tf_gpu
from common.model_utils import get_optimizer
from common.callbacks import CheckpointCleanCallBack

import tensorflow as tf
optimize_tf_gpu(tf, K)


def main(args):
    log_dir = os.path.join('logs', '000')
    class_names = get_classes(args.classes_path)
    assert class_names[0] == 'background', '1st class should be background.'
    num_classes = len(class_names)

    # callbacks for training process
    logging = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=False, write_grads=False, write_images=False, update_freq='batch')
    checkpoint = ModelCheckpoint(os.path.join(log_dir, 'ep{epoch:03d}-loss{loss:.3f}-accuracy{accuracy:.3f}-val_loss{val_loss:.3f}-val_accuracy{val_accuracy:.3f}.h5'),
        monitor='val_accuracy',
        mode='max',
        verbose=1,
        save_weights_only=False,
        save_best_only=True,
        period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, mode='min', patience=10, verbose=1, cooldown=0, min_lr=1e-10)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=1, mode='min')
    checkpoint_clean = CheckpointCleanCallBack(log_dir, max_keep=5)
    terminate_on_nan = TerminateOnNaN()

    callbacks = [logging, checkpoint, reduce_lr, early_stopping, terminate_on_nan, checkpoint_clean]

    # load & update audio params
    if args.params_path:
        inject_params(args.params_path)

    # get train & val dataset
    if args.val_data_path:
        x_train, y_train, _, _ = get_dataset(args.train_data_path, class_names)
        x_val, y_val, _, _ = get_dataset(args.val_data_path, class_names)
    else:
        assert args.val_split > 0, 'no val data split.'
        x_train, y_train, x_val, y_val = get_dataset(args.train_data_path, class_names, args.val_split)

    # prepare optimizer
    if args.decay_type:
        callbacks.remove(reduce_lr)
    steps_per_epoch = max(1, len(x_train)//args.batch_size)
    decay_steps = steps_per_epoch * args.epochs
    optimizer = get_optimizer(args.optimizer, args.learning_rate, average_type=None, decay_type=args.decay_type, decay_steps=decay_steps)

    # prepare loss according to loss type
    if args.background_bias:
        weights = [args.background_bias] + [(1.0 - args.background_bias) / (num_classes - 1)] * (num_classes - 1)
        weights = np.array(weights)
        losses = WeightedSparseCategoricalCrossEntropy(weights)
    else:
        losses = SparseCategoricalCrossEntropy()

    # get train model
    model = get_model(args.model_type, num_classes, args.weights_path)
    model.compile(optimizer=optimizer,
                  loss=losses,
                  metrics=['accuracy'])
    model.summary()

    print('Train on {} samples, val on {} samples, with batch size {}.'.format(len(x_train), len(x_val), args.batch_size))
    model.fit(x_train, y_train,
              batch_size=args.batch_size,
              epochs=args.epochs,
              validation_data=(x_val, y_val),
              #validation_split=args.val_split,
              validation_freq=1,
              callbacks=callbacks,
              shuffle=True,
              verbose=1,
              workers=1,
              use_multiprocessing=False,
              max_queue_size=10)

    # Finally store model
    model.save(os.path.join(log_dir, 'trained_final.h5'))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model definition options
    parser.add_argument('--model_type', type=str, required=False, default='simple_cnn',
        help='classifier model type: simple_cnn/simple_gru/simple_lstm, default=%(default)s')
    parser.add_argument('--weights_path', type=str, required=False, default=None,
        help = "Pretrained model/weights file for fine tune")

    # Data options
    parser.add_argument('--train_data_path', type=str, required=True,
        help='path to train dataset')
    parser.add_argument('--val_data_path', type=str, required=False, default=None,
        help='path to val dataset')
    parser.add_argument('--val_split', type=float, required=False, default=0.15,
        help = "validation data persentage in dataset if no val dataset provide, default=%(default)s")
    parser.add_argument('--classes_path', type=str, required=True,
        help='path to class definitions')
    parser.add_argument('--params_path', type=str, required=False, default=None,
        help='path to params json file')

    # Training options
    parser.add_argument('--background_bias', type=float, required=False, default=None,
        help = "background loss bias (0~1) when training. lower values may cause more false positives if set, default=%(default)s")
    parser.add_argument('--batch_size', type=int, required=False, default=512,
        help = "Batch size for train, default=%(default)s")
    parser.add_argument('--optimizer', type=str, required=False, default='adam', choices=['adam', 'rmsprop', 'sgd'],
        help = "optimizer for training (adam/rmsprop/sgd), default=%(default)s")
    parser.add_argument('--learning_rate', type=float, required=False, default=1e-3,
        help = "Initial learning rate, default=%(default)s")
    parser.add_argument('--decay_type', type=str, required=False, default=None, choices=[None, 'cosine', 'exponential', 'polynomial', 'piecewise_constant'],
        help = "Learning rate decay type, default=%(default)s")

    parser.add_argument('--epochs', type=int,required=False, default=100,
        help = "Total training epochs, default=%(default)s")

    args = parser.parse_args()

    main(args)

