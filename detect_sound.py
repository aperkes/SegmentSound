## Code to detect sound in a wav

## Written by Ammon, email perkes.ammon@gmail.com for questions

# Import junk:
import numpy as np
from pydub import AudioSegment
from scipy.io import wavfile as wf
from scipy.signal import butter, lfilter
from matplotlib import pyplot as plt
from scipy.signal import spectrogram
from matplotlib import pyplot as plt
import argparse

import sys,os

# Set some parameters:
# These are in seconds:
SND_THRESH = .025
SIL_THRESH = .25
AMP_THRESH = 100 ## obviously not seconds, pressure I guess
S_DEFAULT = 3600

parser = argparse.ArgumentParser()
parser.add_argument('-i','--wav',default=None,required=True,help='Input .wav file to run detection on')
parser.add_argument('-s','--segment_size',default=S_DEFAULT,required=False,help='Number of seconds per wav segment')
parser.add_argument('-o','--out_dir',default=None,required=False,help='Directory for storing output (Defaults to directory matching the file name)')
parser.add_argument('-t','--ros_start',default=None,required=False,help='ROS time start (seconds) of the original file, if absent will try to grab it from the .wav filename')
parser.add_argument('-c','--chunk_number',default=None,required=False,help='Number of the chunk, if absent will try to grab from the .wav filename')
parser.add_argument('-w','--save_wav',action="store_true",help='Save the clips as a wavform? Include for True')
parser.add_argument('-d','--drop_png',action="store_true",help='Delete the png rather than save? Include to delete')
args = parser.parse_args()

wav_file = args.wav
wav_name = wav_file.split('/')[-1].replace('.wav','')

if args.out_dir is not None:
    out_dir = args.out_dir
else:
    out_dir = './working_dir/' + wav_name + '/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        os.makedirs(out_dir + '/images')
        os.makedirs(out_dir + '/wavs')
if out_dir[-1] != '/':
    out_dir = out_dir + '/'

## So far we don't actually use this...
if args.ros_start is not None:
    ros_start = args.ros_start
else:
    try:
        ros_start = wav_name.split('_')[-2]
    except:
        ros_start = 0

if args.chunk_number is not None:
    chunk_number = args.chunk_number
else:
    try:
        chunk_number = int(wav_name.split('_')[-1])

    except:
        print('Could not recover chunk number from .wav filename. Defaulting to 0. This could be wrong!')
        chunk_number = 0

"""# Import wav (eventually this could run live (as ros?)) 
if len(sys.argv) > 1:
    wav_file = sys.argv[1]
    out_dir = sys.argv[2]
    if out_dir[-1] != '/':
        out_dir = out_dir + '/'
else:
    wav_file = './annotated_wav_190213.wav'
    out_dir = './timed_wavs_short/'
"""

fs,wav_array = wf.read(wav_file)
wav_audio = AudioSegment.from_wav(wav_file)
wav_audio = wav_audio.set_channels(1)
print('Array shape:',wav_array.shape)

## This needs some additional data: 
#S: n seconds per .wav clip
#ROS Start time
#Original file
S = args.segment_size # This should probably be an argument...

OFFSET = args.segment_size * chunk_number
CLIP_SIZE = fs * 5 ## Clip size for long clips in sampling rate / s * n seconds for n samples
## Convert to stereo if needed
if len(wav_array.shape) > 1:
    print('converting to mono')
    wav_array = wav_array[:,0]
    print('New Shape:',wav_array.shape)

# Bandpass filter (see scipy cookbook)
lowcut = 2000
highcut = 20000
order = 9 ## Might have to figure this out
nyq = 0.5 * fs
low = lowcut / nyq
high = highcut / nyq
## Filter
print('filtering, this could take a minute...')
b,a = butter(order, [low,high], btype='band')
filter_array = lfilter(b,a,wav_array)
print('done!')

# Absolute value and clip
print('clipping...')
clip_array = np.abs(filter_array)
clip_array = np.array(clip_array > AMP_THRESH).astype(int)
## Test with sample data:
#wav_array = np.random.randint(0,high=2,size=1000)

if False:
    fig, (ax1,ax2) = plt.subplots(2)
    ax1.plot(clip_array[::1000])
    ax2.plot(wav_array[::1000])
    ax2.plot(filter_array[::1000])
    ax2.axhline(AMP_THRESH,color='r')
    fig.show()

if False:
    import pdb
    pdb.set_trace()
"""
def save_Sxx_old(out_file,wav_audio,fs=48000):
    clean_array = np.array(wav_audio.get_array_of_samples())
    if len(clean_array.shape) > 1:
        print('converting to mono')
        clean_array = clean_array[:,0]
    f,ts,Sxx = spectrogram(clean_array,fs)
    Sxx = np.flipud(Sxx)
    Sxx = np.log(Sxx)
    Sxx = np.clip(Sxx,-5,3)
    plt.imsave(out_file,Sxx,dpi=300)
"""

def save_Sxx(out_file,wav_array,fs=48000):
    f,ts,Sxx = spectrogram(wav_array,fs)
    Sxx = np.flipud(Sxx)
    Sxx = np.log(Sxx)
    Sxx = np.clip(Sxx,-5,3)
    plt.imsave(out_file,Sxx,dpi=300)

def sound_clip(wav_audio,pos_start,pos_stop,fs=48000,offset = 0):
## I could maybe have a part here to split it into manageable chunks...
    #print(wav_array[pos_start:pos_stop + 1])
    start_ms = int(pos_start / fs * 1000) - 100
    if start_ms < 0:
        start_ms = 0
    true_stop_ms = int(pos_stop / fs * 1000) + 100
    if true_stop_ms > len(wav_audio) -1:
        true_stop_ms = len(wav_audio) - 1
## It's better to do this outside the function as a while loop, no benefit from recursively and the RAM can explode
    """
    if true_stop_ms - start_ms > 5500:
        stop_ms = start_ms + 5000
        print('clipped, making another segment')
        pos_clipped = int(stop_ms * fs / 1000)
        sound_clip(wav_audio,pos_clipped,pos_stop,fs)
    else:
        stop_ms = true_stop_ms
        pos_clipped = int(stop_ms * fs / 1000)
    """
    stop_ms = true_stop_ms
    pos_clipped = int(stop_ms * fs / 1000)
    #out_name = './output_wavs/chunk_' + str(pos_start) + '.wav'
## Chunk it for maskrcnn inputs. 
    #start_ms = start_ms - 500
    #stop_ms = start_ms + 5000
    #out_name = './mask_wavs/chunk_' + str(pos_start) + '.wav'
    out_name = 'clip_' + "%09d" % (start_ms + offset * 1000) + '-' + "%09d" % (stop_ms+offset*1000) + '.wav'
    Sxx_name = out_name.replace('wav','png')

    #print(start_ms,stop_ms)
    chunk = wav_audio[start_ms:stop_ms]
    wav_dir = out_dir.replace('images','wavs')
    if args.save_wav:
        chunk.export(wav_dir + out_name, format="wav") #uncomment if you want to save wavs
    #chunk_array = np.array(chunk)
    chunk_array = wav_array[pos_start:pos_clipped]
    if not args.drop_png:
        save_Sxx(out_dir + Sxx_name,chunk_array,fs)

sound_count = 0
silence_count = 0
counting_silence = False
counting_sound = False
sound_block = False

# process at each sample
print('processing sample:')
for sample_idx in range(len(clip_array)):
    sound = clip_array[sample_idx]
    if sound:
        silence_count = 0
        counting_silence = False
        if counting_sound:
            sound_count += 1
            if sound_count == SND_THRESH * fs:
                counting_sound = False
                sound_block = True
        elif sound_count == 0:
            counting_sound = True
            sound_count += 1
##NOTE I should offset this slightly. Because I'm bandpassing, I'm tossing any low frequency, high amplitude stuff at the outset. 
            #possible_start = sample_idx
            possible_start = max(0,sample_idx - int(.5 * 48000)) ## Pad slightly to catch low frequence burried in the noise
    else: ## i.e. silence
        if silence_count == 0:
            possible_stop = sample_idx
        elif silence_count == SIL_THRESH * fs:
            # It's a real stop! 
            sound_count = 0
            counting_sound = False
            if sound_block:
                clip_start = possible_start
                clip_stop = possible_stop
                while possible_stop - clip_start > 0:
## Get the end of the clip
                     clip_stop = min(clip_start + CLIP_SIZE,possible_stop)
                     print('clipping at:',str(clip_start),str(clip_stop)) 
                     sound_clip(wav_audio,clip_start,clip_stop,offset=OFFSET)
                     clip_start = clip_stop
                print('Done clipping! Moving on to next block')
                #print('clipping... at:',str(possible_start),str(possible_stop))
## This should subclip as a loop rather than recursively
                #sound_clip(wav_audio,possible_start,possible_stop)
                sound_block = False
        silence_count += 1
