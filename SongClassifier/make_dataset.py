#! /usr/bin/env python

## Simple code to take audacity annotations, and export annotated segments

## Made by  Ammon Perkes, email perkes.ammon@gmail.com for questions

## Import necessary stuff
import os
import random
from pydub import AudioSegment
from pydub.utils import mediainfo
from scipy.signal import spectrogram
import numpy as np
from matplotlib import pyplot as plt

## There's also a similar method using Scipy, which might be nicer

wav_file = '../annotated_wav_190213.wav'
wav_file_xml = wav_file.split('/')[-1]
train_dir = './data/binary_wavs/train/'
val_dir = './data/binary_wavs/val/'
out_file = 'chunk_key.txt'
annotation_file = '../sabina_labels.txt'

## Make sure you get everything...
fine_to_course_labels = {'burble':'b', 's burble':'b',
                    'chuck':'c', 'chiuck':'c', 'chucks':'c',
                    'rattle':'n',
                    'close to mic':'s','s':'s','s (human noise)':'s', 's weird':'s', 's1':'s', 's2':'s', 's3':'s',
                    'outside bird':'t', 'outside bird (?)':'t', 'Outside bird':'t', 'outside biird':'t', 'outside birds':'t',
                    's overlap':'o', 's overlap (rattle)':'o', 'overlap':'o', 's overlap ':'o',
                    's whistle':'h', 'whistle':'h', 'multiple whistles':'h', 'whistlte':'h',
                    'rustle noise':'w', 'wing rustle':'w', 'wing wrust':'w','wings':'w','wings?':'w',
                    'noise':'n', 'cage noise':'n', 'outside noise':'n', 'train noise':'n', 'train noise begin':'n','siren':'n'}
course_to_int_labels = {
    'b': 0,
    'c': 1,
    's': 2,
    't': 3,
    'o': 4,
    'h': 5,
    'w': 6,
    'n': 7}

course_to_binary = {
    'b': 'not_song',
    'c': 'not_song',
    's': 'song',
    't': 'not_song',
    'o': 'not_song',
    'h': 'not_song',
    'w': 'not_song',
    'n': 'not_song'}

out_dirs = [course_to_binary[k] for k in course_to_binary.keys()]
## Check to make sure all the container directories are there...
for d in out_dirs:
    if d not in os.listdir(val_dir):
        os.mkdir(val_dir + '/' + d)
    if d not in os.listdir(train_dir):
        os.mkdir(train_dir + '/' + d)

course_to_int_binary = {
    'b': 0,
    'c': 0,
    's': 0,
    't': 0,
    'o': 0,
    'h': 0,
    'w': 0,
    'n': 1}

## Open wav file
song_info = mediainfo(wav_file)
s_rate = song_info['sample_rate']

song = AudioSegment.from_wav(wav_file)
## Read in audacity annotation
with open(annotation_file,'r') as annotation_obj:
## for each annotation, export new wav, and write a key with annotations
    for line in annotation_obj:
## I can't split this by space in this case, because of annotations
        line_split = line.strip().split('\t')
        start,stop,annot = line_split
        if start == stop:
            start = float(start) - 1
            stop = float(stop) + 1
            continue
        annot_course = fine_to_course_labels[annot]
        annot_int = str(course_to_int_labels[annot_course])
        start_hz = int(float(start) * int(s_rate))
        stop_hz = int(float(stop) * int(s_rate))
        start_ms = float(start) * 1000
        stop_ms = float(stop) * 1000
        chunk = song[start_ms:stop_ms]
        file_name = 'chunk_' + str("%09.0f"%start_ms).replace('.','-') + '.png'
        chunk_array = np.array(chunk.get_array_of_samples())
        f,ts,Sxx = spectrogram(chunk_array,int(s_rate))
        Sxx = np.flipud(Sxx)
        Sxx = np.log(Sxx)
        Sxx = np.clip(Sxx,-5,3)
        annot_binary = course_to_binary[annot_course]
        if random.random() < .75:
## This needs to be a spect
            plt.imsave(train_dir + annot_binary + '/' + file_name,Sxx,dpi=300)
        else:
            plt.imsave(val_dir + annot_binary + '/' + file_name,Sxx,dpi=300)


