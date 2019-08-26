## Code to detect sound in a wav

## Written by Ammon, email perkes.ammon@gmail.com for questions

# Import junk:
import numpy as np

# Set some parameters:
SND_THRESH = 5
SIL_THRESH = 3

# Import wav (eventually this could run live) 

# Bandpass filter

# Clip

## Test with sample data:
# High is non-inclusive, so for 1s, you want 2)
wav_array = np.random.randint(0,high=2,size=1000)

def sound_clip(wav_array,pos_start,pos_stop):
## I could maybe have a part here to split it into manageable chunks...
    print(wav_array[pos_start:pos_stop + 1])

sound_count = 0
silence_count = 0
counting_silence = False
counting_sound = False
sound_block = False
# process at each sample
for sample_idx in range(len(wav_array)):
    sound = wav_array[sample_idx]
    if sound:
        silence_count = 0
        counting_silence = False
        if counting_sound:
            sound_count += 1
            if sound_count == SND_THRESH:
                counting_sound = False
                sound_block = True
        elif sound_count == 0:
            counting_sound = True
            sound_count += 1
            possible_start = sample_idx
    else: ## i.e. silence
        if silence_count == 0:
            possible_stop = sample_idx
        elif silence_count == SIL_THRESH:
            # It's a real stop! 
            sound_count = 0
            counting_sound = False
            if sound_block:
                sound_clip(wav_array,possible_start,possible_stop)
        silence_count += 1
