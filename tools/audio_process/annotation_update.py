#!/usr/bin/python3
# -*- coding=utf-8 -*-
import os, sys, argparse
import glob
import json
from tqdm import tqdm
import wave


def get_wav_time(wav_file):
    wf = wave.open(wav_file, 'rb')
    wav_time = wf.getnframes() / wf.getframerate()
    wav_time = round(wav_time, 2)
    wf.close()

    return wav_time


def annotation_update(old_annotation_file, splited_wav_path, output_annotation_file):
    f = open(old_annotation_file)
    annotations = f.readlines()

    output_annotation_fp = open(output_annotation_file, 'w')

    pbar = tqdm(total=len(annotations), desc='Annotation convert')
    for annotation in annotations:
        annotation_data = json.loads(annotation)
        file_path = annotation_data["audio_filepath"]
        file_name = os.path.basename(file_path)

        splited_file_path = os.path.join(splited_wav_path, file_name)
        if os.path.exists(splited_file_path):
            new_annotation = {}
            new_annotation["audio_filepath"] = 'wavs/' + file_name
            new_annotation["duration"] = get_wav_time(splited_file_path)
            new_annotation["text"] = annotation_data["text"][4:] # strip wakeword "nihaoshitou"

            new_annotation_str = json.dumps(new_annotation, ensure_ascii=False)
            output_annotation_fp.write(new_annotation_str)
            output_annotation_fp.write('\n')
        pbar.update(1)
    pbar.close()
    output_annotation_fp.close()


def main():
    parser = argparse.ArgumentParser(description='Convert legacy json annotation with new stripped speech command wav audios')
    parser.add_argument('--old_annotation_file', type=str, required=True,
                        help='wav file or directory to check')
    parser.add_argument('--splited_wav_path', type=str, required=True,
                        help='wav file or directory to check')
    parser.add_argument('--output_annotation_file', type=str, required=True,
                        help='wav file or directory to check')

    args = parser.parse_args()


    annotation_update(args.old_annotation_file, args.splited_wav_path, args.output_annotation_file)



if __name__ == "__main__":
    main()
