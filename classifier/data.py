#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys
import numpy as np
import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from shutil import rmtree
import uuid

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
from common.data_utils import get_mfcc_feature


def get_sample_list(audio_path, class_names):
    sample_list = []

    for class_name in class_names:
        class_path = os.path.join(audio_path, class_name)
        if not os.path.isdir(class_path):
            raise Exception('audio path for \'' + class_name + '\' not found at ' + class_path + '!')

        audio_files = glob.glob(os.path.join(class_path, '*.wav'))
        for audio_file in audio_files:
            sample_list.append({'file': audio_file, 'word': class_name})

    return sample_list


def extract_features(audio_path, class_names):
    """
    get audio samples from path and extract feature vector data
    """
    features = []
    print('Extracting mfcc feature from audio files')

    sample_list = get_sample_list(audio_path, class_names)
    pbar = tqdm(total=len(sample_list), desc='Extracting features')
    for sample in sample_list:
        pbar.update(1)
        mfcc_feature = get_mfcc_feature(sample['file'])

        features.append({'data': mfcc_feature, 'label': sample['word']})
    pbar.close()

    return features


def save_features(features, feature_path):
    """
    save feature vector data to .npy file
    """
    # clean exist feature path
    if os.path.isdir(feature_path):
        rmtree(feature_path)
        os.makedirs(feature_path, exist_ok=True)

    print('Saving mfcc features as npy files to {}'.format(feature_path))
    pbar = tqdm(total=len(features), desc='Saving mfcc features')
    for feature in features:
        pbar.update(1)
        class_path = os.path.join(feature_path, feature['label'])
        os.makedirs(class_path, exist_ok=True)
        file_name = uuid.uuid4().hex + '.npy'
        file_path = os.path.join(class_path, file_name)

        np.save(file_path, feature['data'].astype(np.float32))
    pbar.close()


def split_data(x, y, val_split):
    """
    split feature data and label to train/val set
    """
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=val_split, shuffle=True)

    return np.asarray(x_train), np.asarray(y_train), np.asarray(x_val), np.asarray(y_val)


def get_dataset(dataset_path, class_names, val_split=None):
    """
    load audio data and extract & save feature vectors, then split to train/val set
    """
    # "sounds" dir contains audio sample files, and
    # "features" dir store feature vector data in .npy file
    audio_path = os.path.join(dataset_path, 'sounds')
    feature_path = os.path.join(dataset_path, 'features')

    if os.path.exists(feature_path):
        print('feature files path {} already exists, ignore feature extraction'.format(feature_path))
    else:
        features = extract_features(audio_path, class_names)
        save_features(features, feature_path)
        # destroy feature list object during extraction to save memory
        del(features)

    print('Loading mfcc features into memory')
    x = []
    y = []

    feature_files = glob.glob(os.path.join(feature_path, '*', '*.npy'))
    pbar = tqdm(total=len(feature_files), desc='Loading feature files')
    for feature_file in feature_files:
        pbar.update(1)
        feature_data = np.load(feature_file).astype(np.float32)

        # parse word class name from feature file path
        _, class_name = os.path.split(os.path.dirname(feature_file))
        class_name = class_name.lower()
        label = class_names.index(class_name)

        x.append(feature_data)
        y.append(label)
    pbar.close()

    if val_split:
        x_train, y_train, x_val, y_val = split_data(x, y, val_split)
        return x_train, y_train, x_val, y_val
    else:
        return np.asarray(x), np.asarray(y), None, None

