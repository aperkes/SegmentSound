#! /usr/bin/env python

## (barely Pseudo) code to take a wav + annotations and spit out a training set

import numpy as np
from scipy.signal import spectrogram
from scipy.io import wavform as wf
import json

## Code to make and return spectrogram, obviously the parameters here matter ALOT

VMIN = 0
VMAX = 4

def make_outname(t,prefix='output_',suffix='.png',n_zs = 6):
    out_string = prefix + str(t).zfill(n_zs) + '.png'

def make_spect(wav_chunk,t,save_me = True,vmin=VMIN,vmax=VMAX):
    f,t,Sxx = spectrogram(wav_chunk)
    Sxx = np.log(Sxx)
    if save_me:
        out_string = make_outname(t)
        plt.imsave(out_string,Sxx,vmax=vmax,vmin=vmin)
    return Sxx,t

## Convert t (an index) to time in seconds. I know this is trivial, but I forget things
def sample_to_sec(i,sr=48000):
    t_s = i / sr
    return t_s

def sec_to_sample(t_s,sr=48000):
    i = int(sr * t_s)
    return i

def time_to_pixel(t0,sxx_t):
    p = np.argmax(sxx_t >= t0)
    return p
    
def make_annot(wav_chunk,i,a0,annotations,step_size,skip_list = []):
    sxx,sxx_times,out_file = make_spect(wav_chunk,i,save_me=True)
    t_s = sample_to_sec(i)
    step_size_s = sample_to_sec(step_size)
    sub_annots = annotations[annotations[:,1] > t_s & annotations[:,0] <= (t_s+step_size_s)
    annot_list = []
## process each line and add it to the annotation list
    a_i = a0
    for a in sub_annots:
        annot_dict = {}
        offset_start = sub_annots[0] - t_s
        offset_stop = sub_annots[1] - t_s
        label = sub_annots[2]
        label_id = label_dict[label]
        if label_id in skip_list:
            continue
        x0 = time_to_pixel(offset_start,sxx_times)
        x1 = time_to_pixel(offset_stop,sxx_times)
        y0 = 0
        y1 = sxx.shape[1]-1
        bbox = [x0,x1,y0,y1]
        annot_dict["id"] = a_i
        annot_dict["image_id"] = i
        annot_dict['category_id'] = label_id
        annot_dict['bbox'] = bbox 
        annot_dict["area"] = (y1-y0) * (x1 - x0)
        annot_dict["iscrowd"] = 0
        annot_list.append(annot_dict)
        a_i += 1
# build image meta
    image_meta = {
        "id":i,
        "license":1,
        "file_name":out_file,
        "height":sxx.shape[1],
        "width":sxx.shape[0],
        "date_captured":null
    }
    
    return annot_list,image_meta

## Open .wav/.wv (Can I hold it all in Ram? If not, it would be better to read through it.
sr, data = wf.read('./input_wav.wav')

if len(data.shape) > 1: ## Check if it's stereo
    wav_array = data[0] ## Eventually we should do this for all channels

## Doing all channels could be done as a for loop where we align to the 
##  channels used in the wav

## Open annotations file
annotations = np.from_csv('./annotations_file.txt',delimiter = ',')
## Read through wav, chop into overlapping segments

## Could be flexible, depends on Hz samplring rate
window_size = 5 * 48000

step_size = int(window_size/2)

## Inititalize the annotation dict
annotation_dict = {
    "info":{
        "year":"2021",
        "version":"1.0",
        "description":"Segmented Cowbird Sounds",
        "url":"aperkes.github.io",
        "date_created":"2021-10-20"
    },
    "licenses":[],
    "categories":[],
    "images":[],
    "annotations":[]
}
null_license = {
    "url":"aperkes.github.io",
    "id":None,
    "name":"Not Licenses"
}
a0 = 0

## Step through each img bin and grab the relevent annotations
for i in np.arange(0,len(wav_array),step_size)):
    wav_chunk = wav_array[i:i+window_size]
    #_,t = make_spect(wav_chunk,i,save_me=True) # Do this inside make_annot
    chunk_dicts,img_meta = make_annot(wav_chunk,i,a0,annotations,step_size)
   
    annotation_dict["images"].append(img_meta)
    null_license["id"] = i
    annotation_dict["licenses"].append(null_license)
    for a in chunk_dicts:
        annotation_dict["annotations"].append(a)
        a0 += 1 ## We have to keep track of the annotation id unfortunately
         
## Write annotations to a json
with open("annotations_test.json",'w') as outfile:
    json.dump(annotation_dict,outfile)
    
