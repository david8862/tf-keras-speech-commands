#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
from classifier.models.cnn import SimpleCNN, SimpleCNNLite
from classifier.models.rnn import SimpleGRU, SimpleLSTM
from classifier.params import pr


def get_model(model_type, num_classes, weights_path=None):

    # RNN model use 2D input, while CNN use 3D
    if model_type in ['simple_gru', 'simple_lstm']:
        input_tensor = Input(shape=(pr.n_features, pr.feature_size), name='feature_input')
    else:
        input_tensor = Input(shape=(pr.n_features, pr.feature_size, 1), name='feature_input')


    if model_type == 'simple_cnn':
        base_model = SimpleCNN(input_tensor=input_tensor, input_shape=(pr.n_features, pr.feature_size, 1), feature_size=128, include_top=False)
    elif model_type == 'simple_cnn_lite':
        base_model = SimpleCNNLite(input_tensor=input_tensor, input_shape=(pr.n_features, pr.feature_size, 1), feature_size=128, include_top=False)
    elif model_type == 'simple_gru':
        base_model = SimpleGRU(input_tensor=input_tensor, input_shape=(pr.n_features, pr.feature_size), recurrent_units=48, include_top=False)
    elif model_type == 'simple_lstm':
        base_model = SimpleLSTM(input_tensor=input_tensor, input_shape=(pr.n_features, pr.feature_size), recurrent_units=48, include_top=False)
    else:
        raise ValueError('Unsupported model type')

    x = base_model.output

    # and a logistic layer
    predictions = Dense(num_classes, activation='softmax', name='score_predict')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    if weights_path:
        model.load_weights(weights_path, by_name=False)#, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))

    return model

