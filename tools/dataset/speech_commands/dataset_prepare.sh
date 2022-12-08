#!/bin/bash
#
# prepare Google Speech Commands dataset for wake word detection
#
# Project link:
# https://www.tensorflow.org/datasets/catalog/speech_commands
#
if [[ "$#" -ne 1 ]]; then
    echo "Usage: $0 <wakeword>"
    exit 1
fi

WAKE_WORD=$1

# Google Speech Commands dataset v0.02, 2.3GB
echo "Downloading Google Speech Commands dataset v0.02..."
wget http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz

# Google Speech Commands mini dataset, 174MB
echo "Downloading Google Speech Commands mini dataset..."
wget http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip

# extract speech commands data
echo "Extract speech commands data..."
mkdir -p raw_data && tar xzf speech_commands_v0.02.tar.gz -C raw_data
rm -rf raw_data/_background_noise_

# prepare folder & pick test samples from val & test list
mkdir -p $WAKE_WORD
pushd $WAKE_WORD
mkdir -p wake-word not-wake-word test/wake-word test/not-wake-word
popd

echo "Prepare test samples..."
cat raw_data/validation_list.txt raw_data/testing_list.txt | while read line
do
    #speech_command=`echo $line | cut -d "/" -f1`
    #speech_command=`echo $line | awk '{split($1, arr, "/"); print arr[1]}'`
    speech_info=(${line//\// })
    speech_command=${speech_info[0]}
    speech_file=${speech_info[1]}

    if [ "$speech_command" == "$WAKE_WORD" ]; then
        mv raw_data/$line $WAKE_WORD/test/wake-word/${speech_command}_${speech_file}
    else
        # different speech command may have audio sample with same name, so here we
        # add a "speech_command" prefix
        mv raw_data/$line $WAKE_WORD/test/not-wake-word/${speech_command}_${speech_file}
    fi
done

echo "Prepare train samples..."
pushd raw_data
speech_file_list=$(find -name *.wav)
for line in $speech_file_list
do
    speech_info=(${line//\// })
    speech_command=${speech_info[1]}
    speech_file=${speech_info[2]}

    if [ "$speech_command" == "$WAKE_WORD" ]; then
        mv $line ../$WAKE_WORD/wake-word/${speech_command}_${speech_file}
    else
        # add "speech_command" prefix
        mv $line ../$WAKE_WORD/not-wake-word/${speech_command}_${speech_file}
    fi
done
popd


# clean up
echo "Clean up..."
rm -rf raw_data
echo "Done"
