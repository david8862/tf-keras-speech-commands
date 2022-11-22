#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow.keras.backend as K


class SparseCategoricalCrossEntropy(object):
    """
    compute sparse categorical cross entropy, with ignore_index support.
    Reference from:
        https://github.com/keras-team/keras/issues/6118
    """
    def __init__(self, ignore_index=None, from_logits=False):
        self.ignore_index = ignore_index
        self.from_logits = from_logits
        self.__name__ = 'sparse_categorical_crossentropy'

    def __call__(self, y_true, y_pred):
        return self.sparse_categorical_crossentropy(y_true, y_pred)

    def sparse_categorical_crossentropy(self, y_true, y_pred):
        num_classes = K.shape(y_pred)[-1]

        # check y_true to get label keep mask
        if self.ignore_index:
            label_mask = K.cast(K.not_equal(y_true, self.ignore_index), 'float32')
            #y_true = y_true * label_mask

        y_true = K.one_hot(K.cast(y_true[..., 0], 'int32'), num_classes)
        y_true = K.cast(y_true, 'float32')
        y_pred = K.cast(y_pred, 'float32')

        if self.from_logits:
            y_pred = K.softmax(y_pred)

        losses = K.categorical_crossentropy(y_true, y_pred)

        # apply label keep mask to ignore index
        if self.ignore_index:
            losses *= K.squeeze(label_mask, axis=-1)

        return losses


class WeightedSparseCategoricalCrossEntropy(object):
    def __init__(self, weights, ignore_index=None, from_logits=False):
        self.weights = np.array(weights).astype('float32')
        self.ignore_index = ignore_index
        self.from_logits = from_logits
        self.__name__ = 'weighted_sparse_categorical_crossentropy'

    def __call__(self, y_true, y_pred):
        return self.weighted_sparse_categorical_crossentropy(y_true, y_pred)

    def weighted_sparse_categorical_crossentropy(self, y_true, y_pred):
        num_classes = len(self.weights)

        # check y_true to get label keep mask
        if self.ignore_index:
            label_mask = K.cast(K.not_equal(y_true, self.ignore_index), 'float32')
            #y_true = y_true * label_mask

        y_true = K.one_hot(K.cast(y_true[..., 0], 'int32'), num_classes)
        if self.from_logits:
            y_pred = K.softmax(y_pred)

        log_pred = K.log(y_pred)
        unweighted_losses = -K.sum(y_true*log_pred, axis=-1)

        weights = K.sum(K.constant(self.weights) * y_true, axis=-1)
        weighted_losses = unweighted_losses * weights

        # apply label keep mask to ignore index
        if self.ignore_index:
            weighted_losses *= K.squeeze(label_mask, axis=-1)

        return weighted_losses

