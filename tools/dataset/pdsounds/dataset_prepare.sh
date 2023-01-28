#!/bin/bash
#
# prepare Public Domain Sounds Backup dataset
#
# Project link:
# http://pdsounds.tuxfamily.org/
#

# Public Domain Sounds Backup data, 525MB
echo "Downloading Public Domain Sounds Backup data..."
wget http://downloads.tuxfamily.org/pdsounds/pdsounds_march2009.7z

# extract & convert data
echo "Extract & convert data..."
apt install p7zip
7zr x pdsounds_march2009.7z -opdsounds

python ../../audio_process/audio_convert.py --audio_path=./pdsounds/mp3 --output_path=./pdsounds/wav --channel_num=1 --sample_rate=16000 --sample_bit=16 --target_format=wav

echo "Done"
