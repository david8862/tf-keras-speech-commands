#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys

from tensorflow.keras.layers import Conv2D, SeparableConv2D, MaxPooling2D, BatchNormalization, ReLU, Dropout, Flatten, Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K


def SimpleCNN(input_shape=None,
              input_tensor=None,
              feature_size=128,
              include_top=False,
              classes=1000,
              dropout_rate=0.5,
              **kwargs):
    # If input_shape is None and input_tensor is None using standard shape
    if input_shape is None and input_tensor is None:
        input_shape = (None, None, 1)

    if input_tensor is None:
        feature_input = Input(shape=input_shape)
    else:
        feature_input = input_tensor

    x = Conv2D(filters=16,
               kernel_size=3,
               strides=1,
               padding='same',
               use_bias=False)(feature_input)
    x = BatchNormalization()(x)
    x = ReLU(6.)(x)
    x = MaxPooling2D()(x)

    x = Conv2D(filters=32,
               kernel_size=3,
               strides=1,
               padding='same',
               use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU(6.)(x)
    x = MaxPooling2D()(x)

    x = Conv2D(filters=64,
               kernel_size=3,
               strides=2,
               padding='same',
               use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU(6.)(x)

    x = Conv2D(filters=128,
               kernel_size=3,
               activation='relu',
               strides=1,
               padding='same',
               use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU(6.)(x)
    x = MaxPooling2D()(x)

    x = Flatten()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(feature_size, use_bias=True)(x)
    x = ReLU(6.)(x)

    if include_top:
        x = Dense(classes, activation='softmax')(x)

    # create model
    model = Model(feature_input, x)

    return model


def SimpleCNNLite(input_shape=None,
                  input_tensor=None,
                  feature_size=128,
                  include_top=False,
                  classes=1000,
                  dropout_rate=0.5,
                  **kwargs):
    # If input_shape is None and input_tensor is None using standard shape
    if input_shape is None and input_tensor is None:
        input_shape = (None, None, 1)

    if input_tensor is None:
        feature_input = Input(shape=input_shape)
    else:
        feature_input = input_tensor

    x = SeparableConv2D(filters=16,
                        kernel_size=3,
                        strides=1,
                        padding='same',
                        use_bias=True)(feature_input)
    x = BatchNormalization()(x)
    x = ReLU(6.)(x)
    x = MaxPooling2D()(x)

    x = SeparableConv2D(filters=32,
                        kernel_size=3,
                        strides=1,
                        padding='same',
                        use_bias=True)(x)
    x = BatchNormalization()(x)
    x = ReLU(6.)(x)
    x = MaxPooling2D()(x)

    x = SeparableConv2D(filters=64,
                        kernel_size=3,
                        activation='relu',
                        strides=2,
                        padding='same',
                        use_bias=True)(x)
    x = BatchNormalization()(x)
    x = ReLU(6.)(x)

    x = SeparableConv2D(filters=128,
                        kernel_size=3,
                        activation='relu',
                        strides=1,
                        padding='same',
                        use_bias=True)(x)
    x = BatchNormalization()(x)
    x = ReLU(6.)(x)
    x = MaxPooling2D()(x)

    x = Flatten()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(feature_size, use_bias=True)(x)
    x = ReLU(6.)(x)

    if include_top:
        x = Dense(classes, activation='softmax')(x)

    # create model
    model = Model(feature_input, x)

    return model
