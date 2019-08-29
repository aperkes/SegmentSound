## Code to detect sound in a wav

## Written by Ammon, email perkes.ammon@gmail.com for questions

# Import junk:
import numpy as np
from pydub import AudioSegment
from scipy.io import wavfile as wf
from scipy.signal import butter, lfilter
from matplotlib import pyplot as plt

# Set some parameters:
# These are in seconds:
SND_THRESH = .025
SIL_THRESH = .25
AMP_THRESH = 100
# Import wav (eventually this could run live (as ros?)) 
wav_file = './annotated_wav_190213.wav'
fs,wav_array = wf.read(wav_file)
wav_audio = AudioSegment.from_wav(wav_file)

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

def sound_clip(wav_array,pos_start,pos_stop,fs=48000):
## I could maybe have a part here to split it into manageable chunks...
    #print(wav_array[pos_start:pos_stop + 1])
    start_ms = int(pos_start / fs * 1000)
    stop_ms = int(pos_stop / fs * 1000)
    #out_name = './output_wavs/chunk_' + str(pos_start) + '.wav'
## Chunk it for maskrcnn inputs. 
    start_ms = start_ms - 500
    stop_ms = start_ms + 5000
    out_name = './mask_wavs/chunk_' + str(pos_start) + '.wav'

    chunk = wav_array[start_ms:stop_ms]
    chunk.export(out_name, format="wav")

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
            possible_start = sample_idx
    else: ## i.e. silence
        if silence_count == 0:
            possible_stop = sample_idx
        elif silence_count == SIL_THRESH * fs:
            # It's a real stop! 
            sound_count = 0
            counting_sound = False
            if sound_block:
                print('clipping... at:',str(possible_start),str(possible_stop))
                sound_clip(wav_audio,possible_start,possible_stop)
                sound_block = False
        silence_count += 1
