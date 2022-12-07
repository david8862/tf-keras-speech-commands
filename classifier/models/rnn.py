#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys
from tensorflow.keras.layers import Input, GRU, LSTM, Dense
from tensorflow.keras.models import Model

import tensorflow.keras.backend as K


def SimpleGRU(input_shape=None,
              input_tensor=None,
              recurrent_units=48,
              include_top=False,
              classes=1000,
              dropout_rate=0.2,
              **kwargs):
    # If input_shape is None and input_tensor is None using standard shape
    if input_shape is None and input_tensor is None:
        input_shape = (None, None)

    if input_tensor is None:
        feature_input = Input(shape=input_shape)
    else:
        feature_input = input_tensor

    x = GRU(recurrent_units, activation='linear',
            dropout=dropout_rate, name='gru_unit')(feature_input)

    if include_top:
        x = Dense(classes, activation='softmax')(x)

    # create model
    model = Model(feature_input, x)

    return model


def SimpleLSTM(input_shape=None,
               input_tensor=None,
               recurrent_units=48,
               include_top=False,
               classes=1000,
               dropout_rate=0.2,
               **kwargs):
    # If input_shape is None and input_tensor is None using standard shape
    if input_shape is None and input_tensor is None:
        input_shape = (None, None)

    if input_tensor is None:
        feature_input = Input(shape=input_shape)
    else:
        feature_input = input_tensor

    x = LSTM(recurrent_units, activation='tanh',
            dropout=dropout_rate, name='lstm_unit')(feature_input)

    if include_top:
        x = Dense(classes, activation='softmax')(x)

    # create model
    model = Model(feature_input, x)

    return model

