#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys
from tensorflow.keras.layers import Input, GRU, LSTM, Dense
from tensorflow.keras.models import Model

import tensorflow.keras.backend as K


def SimpleGRU(input_shape=None,
              input_tensor=None,
              recurrent_units=48,
              num_layers=1,
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

    x = feature_input
    if num_layers > 1:
        for i in range(num_layers-1):
            # mid layer gru need to return sequences
            x = GRU(recurrent_units, activation='linear', return_sequences=True,
                    dropout=dropout_rate, name='gru_unit_'+str(i))(x)

    x = GRU(recurrent_units, activation='linear',
            dropout=dropout_rate, name='gru_unit_'+str(num_layers-1))(x)

    if include_top:
        x = Dense(classes, activation='softmax')(x)

    # create model
    model = Model(feature_input, x)

    return model


def SimpleLSTM(input_shape=None,
               input_tensor=None,
               recurrent_units=48,
               num_layers=1,
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

    x = feature_input
    if num_layers > 1:
        for i in range(num_layers-1):
            # mid layer lstm need to return sequences
            x = LSTM(recurrent_units, activation='tanh', return_sequences=True,
                     dropout=dropout_rate, name='lstm_unit_'+str(i))(x)

    x = LSTM(recurrent_units, activation='tanh',
             dropout=dropout_rate, name='lstm_unit_'+str(num_layers-1))(x)

    if include_top:
        x = Dense(classes, activation='softmax')(x)

    # create model
    model = Model(feature_input, x)

    return model


if __name__ == '__main__':
    input_tensor = Input(shape=(39, 13), batch_size=1, name='feature_input')
    model = SimpleGRU(input_tensor=input_tensor,
                      recurrent_units=48,
                      num_layers=2,
                      include_top=True,
                      classes=2)

    model.summary()
    K.set_learning_phase(0)
