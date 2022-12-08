#!/bin/bash
#
# convert mp3 audio to wav audio with 16K sample rate,
# 16-bit little endian sample width and single channel
#
# need to install ffmpeg with
# $ apt install ffmpeg
#

if [[ "$#" -ne 2 ]]; then
    echo "Usage: $0 <source_path> <dest_path>"
    exit 1
fi

SOURCE_PATH=$1
DEST_PATH=$2

AUDIO_FILE_LIST=$(ls $SOURCE_PATH/*.mp3)
AUDIO_FILE_NUM=$(ls $SOURCE_PATH/*.mp3 | wc -l)

# prepare process bar
i=0
ICON_ARRAY=("\\" "|" "/" "-")

mkdir -p $DEST_PATH

for AUDIO_FILE in $SOURCE_PATH/*.mp3
do
    FILE_NAME=${AUDIO_FILE##*/}
    ffmpeg -i "$AUDIO_FILE" -acodec pcm_s16le -ar 16000 -ac 1 -f wav "$DEST_PATH/${FILE_NAME%.*}.wav" -loglevel quiet -y 2>&1 >> /dev/null

    # update process bar
    let index=i%4
    let percent=i*100/AUDIO_FILE_NUM
    let num=percent/2
    bar=$(seq -s "#" $num | tr -d "[:digit:]")
    #printf "convert process: %d/%d [%c]\r" "$i" "$IMAGE_NUM" "${ICON_ARRAY[$index]}"
    printf "convert process: %d/%d [%-50s] %d%% \r" "$i" "$AUDIO_FILE_NUM" "$bar" "$percent"
    let i=i+1
done
printf "\nDone\n"
