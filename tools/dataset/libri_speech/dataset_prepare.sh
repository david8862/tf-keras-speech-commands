#!/bin/bash
#
# prepare LibriSpeech English ASR corpus dataset
#
# Project link:
# https://www.openslr.org/12
#

# LibriSpeech 100 hours training set clean speech data, 6.3GB
# Alternative mirror link:
# US: https://us.openslr.org/resources/12/train-clean-100.tar.gz
# EU: https://openslr.elda.org/resources/12/train-clean-100.tar.gz
# CN: https://openslr.magicdatatech.com/resources/12/train-clean-100.tar.gz
echo "Downloading LibriSpeech training set data..."
wget https://www.openslr.org/resources/12/train-clean-100.tar.gz

# LibriSpeech development set, clean speech data, 337MB
# Alternative mirror link:
# US: https://us.openslr.org/resources/12/dev-clean.tar.gz
# EU: https://openslr.elda.org/resources/12/dev-clean.tar.gz
# CN: https://openslr.magicdatatech.com/resources/12/dev-clean.tar.gz
echo "Downloading LibriSpeech development data..."
wget https://www.openslr.org/resources/12/dev-clean.tar.gz


# extract & convert training data
echo "Extract & convert training data..."
tar xzvf train-clean-100.tar.gz
mkdir raw_train_corpus train_corpus
pushd LibriSpeech/train-clean-100
find -name *.flac | xargs -n1 -i mv {} ../../raw_train_corpus
popd

python ../../audio_process/audio_convert.py --audio_path=./raw_train_corpus --output_path=./train_corpus --channel_num=1 --sample_rate=16000 --sample_bit=16 --target_format=wav


# extract & convert development data
echo "Extract & convert development data..."
tar xzvf dev-clean.tar.gz
mkdir raw_dev_corpus dev_corpus
pushd LibriSpeech/dev-clean
find -name *.flac | xargs -n1 -i mv {} ../../raw_dev_corpus
popd

python ../../audio_process/audio_convert.py --audio_path=./raw_dev_corpus --output_path=./dev_corpus --channel_num=1 --sample_rate=16000 --sample_bit=16 --target_format=wav


# clean up
echo "Clean up..."
rm -rf raw_train_corpus raw_dev_corpus LibriSpeech
echo "Done"
