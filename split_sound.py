#! /usr/bin/env python

## code to take a wav + annotations and spit out a training set
## Hastily put together and probably a bit buggy
## for use in Marc Schmidt's lab
## For questions, contact Ammon Perkes: perkes.ammon@gmail.com

import json

import numpy as np
import scipy
import scipy.io
from matplotlib import pyplot as plt
from scipy.io import wavfile as wf
from scipy.signal import spectrogram

## Code to make and return spectrogram, obviously the parameters here matter ALOT

VMIN = -3 
VMAX = 3 
CMAP = 'viridis'

all_dir = './data/5_sec_wavs/'



def make_spect(wav_chunk,c,image_name,fs=48000,save_me = True,vmin=VMIN, vmax=VMAX,cmap=CMAP):
    f,t,Sxx = spectrogram(wav_chunk,fs,window="hann",nperseg=512,noverlap=384)
    Sxx = np.log(Sxx)
    Sxx = np.flipud(Sxx)
    Sxx = np.clip(Sxx,vmin,vmax)
    #out_string = make_outname(filename,c)
    #print(Sxx.shape)
    if save_me:
        #plt.imsave(out_string,Sxx)#,vmin=vmin,vmax=vmax)
        plt.imsave(all_dir + '/' + image_name,Sxx,cmap=cmap) # save images to folder
    return Sxx,t, image_name

## n_za is number of zeros, c is the channel
def make_outname(filename,c='',prefix='output_',n_zs = 6):
    out_string = prefix + str(int(filename)).zfill(n_zs) + '_'+ str(c) + '.png'
    return out_string

## Convert t (an index) to time in seconds. I know this is trivial, but I forget things
#sr = sampling rate 

###############################   what is the purpose of these two functions???
def sample_to_sec(i,sr=48000):
    t_s = i / sr
    return t_s

#i is sample
def sec_to_sample(t_s,sr=48000):
    i = int(sr * t_s)
    return i
############################################################

def time_to_pixel(t0,sxx_t):
    #sxx_t = sxx_t / 48000 ## convert sample times to times in seconds
    p = np.argmax(sxx_t >= t0)
    return int(p)
    
def make_annot(wav_chunk,i,c,image_id,a0,annotations,annotations_l,step_size,image_name = None,save_imgs = True,skip_list = []):
    if image_name is None:
        image_name = 'output_' + str(image_id).zfill(6) + '_' + str(c) + '.png'
    t_s = sample_to_sec(i)
    step_size_s = sample_to_sec(step_size)
    
    #window_indices = (np.where(annotations[:,1] > t_s)) and (np.where(annotations[:,0] <= (t_s + window_size / 48000)))
    window_indices = (annotations[:,0] > t_s) & (annotations[:,1] <= (t_s + window_size/48000))

    sub_annots = annotations[window_indices] #annotation event start and stop time within chunk
    sub_labels = annotations_l[window_indices] #annotation event label within chunk

    ## Trim the wav_chunk to exclude overflowing songs
    # Find overlapping 
    if True:

        overlap_left = (annotations[:,1] > t_s) & (annotations[:,0] < t_s )
        overlap_right = (annotations[:,0] <= (t_s + window_size/48000)) & (annotations[:,1] >= (t_s + window_size/48000))
        n_left = np.sum(overlap_left)
        n_right = np.sum(overlap_right)

        new_start_i,new_stop_i = 0,len(wav_chunk)
        new_start = t_s
        new_stop = window_size
        if n_left > 0 or n_right > 0:
            #print('This one!!',image_id)
            if n_left > 0:
                overlap_annots_left = annotations[overlap_left]
                new_start = np.max(overlap_annots_left[:,1]) ## The new start in s
                new_start_i = sec_to_sample(new_start - t_s) ## The new start index for wav_chunk
                
            if n_right > 0:
                overlap_annots_right = annotations[overlap_right]
                new_stop = np.min(overlap_annots_right[:,0])
                new_stop_i = sec_to_sample(new_stop - t_s) ## Get the index of the first annotation
        t_s = new_start
        #print(new_start_i,new_stop_i)
        if new_stop_i - new_start_i < 512:
            sub_annots = []
        else:
            wav_chunk = wav_chunk[new_start_i:new_stop_i]
        sxx,sxx_times,out_file= make_spect(wav_chunk,c,image_name,save_me=save_imgs)
    else:
        sxx,sxx_times,out_file= make_spect(wav_chunk,c,image_name,save_me=save_imgs)
    #print(i)
    #print(t_s)
    #print(wav_chunk)
    #print(sub_annots)
    #print(sub_labels)

    annot_list = []
    label_dict = {
        'song':'s',
        'burble':'b',
        'chatter':'c',
        'chuck':'h',
        'whistle':'w'}

## process each line and add it to the annotation list
    a_i = a0
    for a in range(len(sub_annots)): #loop over for # of annotations within wav chunk
        sub_annot = sub_annots[a]
        annot_dict = {}
        offset_start = sub_annots[a,0] - t_s
        offset_stop = sub_annots[a,1] - t_s
        
        label = sub_labels[a] 

        label_id = label_dict[label]
        if label_id in skip_list:
            continue
        x0 = time_to_pixel(offset_start,sxx_times) #top left x
        x1 = time_to_pixel(offset_stop,sxx_times) #bottom right x
        if x1 == 0:
            x1 = time_to_pixel(sxx_times[-1]/48000,sxx_times) 
        y0 = 0 #top left y
        y1 = int(sxx.shape[0]) #bottom right y

        bbox = [x0,y0,x1,y1]  #top left x, top left y, width, height

        if x0 == x1:
            continue
        print(image_id,a_i,x0,x1,label_id)
        annot_dict["id"] = a_i
        annot_dict["image_id"] = image_id
        annot_dict['category_id'] = label_id #change this to numbers 
        annot_dict['bbox'] = bbox 
        annot_dict["area"] = (y1-y0) * (x1 - x0)
        annot_dict["iscrowd"] = 0
        annot_list.append(annot_dict)
        a_i += 1

# build image meta
    height, width = sxx.shape

    image_meta = {
        "id":image_id,
        "license":1,
        "file_name":out_file,
        "height":height,
        "width":width,
        "date_captured": 'null'
    }
    
    return annot_list,image_meta,a_i

def init_anno_dict():

## Inititalize the annotation dict
    annotation_dict = {
        "info":{
            "year":"2021",
            "version":"1.0",
            "description":"Segmented Cowbird Sounds",
            "url":"aperkes.github.io",
            "date_created":"2021-10-20"
        },
        "licenses":[ {
                "id": 1,
                "name": " ",
                "url": " "
            }],
        "categories":[
            {"supercategory": "bird",
                "id": 0,
                "name": "s"
            },
            {
                "supercategory": "bird",
                "id": 1,
                "name": "c"
            },
            {
                "supercategory": "bird",
                "id": 2,
                "name": "h"
            },
            {
                "supercategory": "bird",
                "id": 3,
                "name": "w"
            },
            {"supercategory": "bird",
                "id": 4,
                "name": "b"}],
        "images":[],
        "annotations":[]
    }
    return annotation_dict


## Open .wav/.wv (Can I hold it all in Ram? If not, it would be better to read through it.  #what is sr - sampling rate
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input_wav',required=False,default='./all_wav440_340.wav')
    parser.add_argument('-l','--input_annotations',required=False,default='./ammon_labels.txt')
    parser.add_argument('-w','--window',default=5,help='number of seconds per window')
    parser.add_argument('-r','--sampling_rate',default=48000)
    parser.add_argument('-j','--annotation_json',default=None,help='Existing json to append new annotations')
    parser.add_argument('-a','--annotation_index',default=0,type=int,help='Start of annotation index from prior json')
    parser.add_argument('-x','--image_index',default=0,type=int,help='Start of image index from prior annotations')
    parser.add_argument('-o','--out_file',default='./annotation_test.json',type=str)
    parser.add_argument('-s','--save_images',action='store_true')
    args = parser.parse_args()
        

    #sr, data = wf.read('./wav440_340.wav')
    file_type = args.input_wav.split('.')[-1]
    if file_type == 'wv':
        raise ValueError('Audio input needs to be .wav. Use ffmpeg to convert')    

    sr, data = wf.read(args.input_wav)
    wav_array = np.transpose(data)

    #print(data.shape)

## Could be flexible, depends on Hz samplring rate
    window_size = 5 * args.sampling_rate

    step_size = int(window_size/2)

    a0 = args.annotation_index
    image_id = args.image_index 
    prefix = args.input_wav.split('/')[-1].replace('.wav','')

## Doing all channels could be done as a for loop where we align to the 
##  channels used in the wav

## Open annotations file
# annotations = np.loadtxt('./ammon_labels.txt',delimiter = '\t')
    annotations = np.loadtxt(args.input_annotations,delimiter = '\t',dtype=str)

    annotations_l = annotations[:,2] # annotation labels
    annotations = annotations[:,:2].astype(float) #annotation event start and stop time 

    if args.annotation_json is None:
        annotation_dict = init_anno_dict()
    else:
        with open(args.annotation_json) as json_file:
            annotation_dict = json.load(json_file)

## Step through each img bin and grab the relevent annotations
## i is the whole file sample number, which jumps by the step size
    n_zs = 5
    for i in np.arange(0,len(wav_array[0]),step_size):
        for c in range(wav_array.shape[0]): # for each channel c
            wav_chunk = wav_array[c,i:i+window_size] #wav_chunk consists of all values within wav-array ranging from i to i+windowsize

            #print(i)
            #print(step_size)
            
            ## This function is a bit of a mess...too late now. It needs everything because it makes the annotations and the images
            start_time = int(i/args.sampling_rate)
            image_name = prefix + '-' + str(start_time).zfill(n_zs) + '_' + str(c).zfill(2) + '.png'
            chunk_dicts,img_meta,a0 = make_annot(wav_chunk,i,c,image_id,a0,annotations,annotations_l,step_size,image_name=image_name,save_imgs=args.save_images)
            annotation_dict["images"].append(img_meta)

            for a in chunk_dicts:
                annotation_dict["annotations"].append(a)

            image_id += 1
            filename = image_id

## Write annotations to a json
        #if i > step_size * 20:
        #    break 
    with open(args.out_file,'w') as outfile:
        json.dump(annotation_dict,outfile)
        
print('done')
