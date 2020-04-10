#! /usr/bin/env python

## Simple code to take audacity annotations, and export annotated segments

## Made by  Ammon Perkes, email perkes.ammon@gmail.com for questions

## Import necessary stuff
from pydub import AudioSegment
from pydub.utils import mediainfo
from xml.etree import ElementTree as ET

## There's also a similar method using Scipy, which might be nicer

wav_file = './annotated_wav_190213.wav'
wav_file_xml = wav_file.split('/')[-1]
out_dir = './inference_wavs'
out_file = 'chunk_key.txt'
annotation_file = './sabina_labels.txt'

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
out_annotation = open(out_dir + out_file,'w')
## Read in audacity annotation
seqs = ET.Element('Sequences')
with open(annotation_file,'r') as annotation_obj:
## for each annotation, export new wav, and write a key with annotations
    for line in annotation_obj:
## I can't split this by space in this case, because of annotations
        line_split = line.strip().split('\t')
        start,stop,annot = line_split
        if start == stop:
            #continue
            start = float(start) - 1
            stop = float(stop) + 1
        annot_course = fine_to_course_labels[annot]
        annot_int = str(course_to_int_labels[annot_course])
        start_hz = int(float(start) * int(s_rate))
        stop_hz = int(float(stop) * int(s_rate))
        start_ms = float(start) * 1000
        stop_ms = float(stop) * 1000
        chunk = song[start_ms:stop_ms]
        file_name = 'chunk_' + str("%09.0f"%start_ms).replace('.','-') + '.wav'
        chunk.export(out_dir + file_name, format="wav")
        out_annotation.write(file_name + ',' + annot_int + '\n')

        ## Parse out the xml code for annotations.xml file
        seq = ET.SubElement(seqs,'Sequence')
        wav_name = ET.SubElement(seq,'WaveFileName')
        wav_name.text = wav_file_xml
        pos = ET.SubElement(seq,'Position')
        pos.text = str(start_hz)
        length = ET.SubElement(seq,'Length')
        length.text = str(stop_hz - start_hz)
        num_note = ET.SubElement(seq,'NumNote')
        num_note.text = str(1)
        note = ET.SubElement(seq,'Note')
        ##NOTE: If we split the song, this is where that happens
        note_pos = ET.SubElement(note,'Position')
        note_pos.text = str(start_hz)
        note_length = ET.SubElement(note,'Length')
        note_length.text = str(stop_hz - start_hz)
        label = ET.SubElement(note,'Label')
        label.text = annot_int

tree_text = ET.tostring(seqs)
with open("annotation.xml",'wb') as tree_file:
    tree_file.write(tree_text)

## That's all! 
out_annotation.close()
