#!/bin/bash
#
# prepare Aishell Mandarin ASR corpus dataset
#
# Project link:
# https://www.openslr.org/33
#

# Aishell 178 100 hours mandarin speech data and transcripts, 15GB
# Alternative mirror link:
# US: https://us.openslr.org/resources/33/data_aishell.tgz
# EU: https://openslr.elda.org/resources/33/data_aishell.tgz
# CN: https://openslr.magicdatatech.com/resources/33/data_aishell.tgz
echo "Downloading Aishell speech data..."
wget https://www.openslr.org/resources/33/data_aishell.tgz


# Aishell supplementary resources, 1.2MB
# Alternative mirror link:
# US: https://us.openslr.org/resources/33/resource_aishell.tgz
# EU: https://openslr.elda.org/resources/33/resource_aishell.tgz
# CN: https://openslr.magicdatatech.com/resources/33/resource_aishell.tgz
echo "Downloading Aishell supplementary resources..."
wget https://www.openslr.org/resources/33/resource_aishell.tgz



# extract & convert training data
echo "Extract & convert speech data..."
tar xzvf data_aishell.tgz
#tar xzvf resource_aishell.tgz

# extract all pakcages
mkdir aishell aishell_wav
ls data_aishell/wav/*.tar.gz | xargs -n1 -i tar -xzvf {} -C aishell
# move wav file together
pushd aishell
find -name *.wav | xargs -n1 -i mv {} ../aishell_wav/
popd

# now the wav audio has been 16 bit & 16k sample rate, so no need to convert
#python ../../audio_process/audio_convert.py --audio_path=./aishell_wav/ --output_path=./aishell_wav2/ --channel_num=1 --sample_rate=16000 --sample_bit=16 --target_format=wav

# split wav to 1 second clips
echo "Split wav audio to 1 second clips..."
python ../../audio_process/audio_split.py --audio_path=./aishell_wav/ --output_path=./aishell_wav_clip/ --split_length=1000 --target_format=wav


# clean up
echo "Clean up..."
rm -rf aishell aishell_wav
echo "Done"
