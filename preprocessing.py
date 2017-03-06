import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib
import sklearn
import librosa
import cmath
import math
from IPython.display import Audio
import glob

def detect_leading_silence(sound, silence_threshold=.001, chunk_size=10):
    # this function first normalizes audio data
    #calculates the amplitude of each frame
    #silence_threshold is used to flip the silence part
    #the number of silence frame is returned.
    #trim_ms is the counter
    trim_ms = 0
    max_num = max(sound)
    sound = sound/max_num
    sound = np.array(sound)
    for i in range(len(sound)):
        if sound[trim_ms] < silence_threshold:
            trim_ms += 1
    return trim_ms

def main():
    sr = 44100
    window_size = 2048
    hop_size = window_size/2

    label = []
    feature = []
    #read file
    for filename in glob.glob('final_data/data/*/*.mp3'):
        music, sr= librosa.load(filename, sr = sr)

        start_trim = detect_leading_silence(music)
        end_trim = detect_leading_silence(np.flipud(music))

        duration = len(music)
        trimmed_sound = music[start_trim:duration-end_trim]
        # the sound without silence

        #use mfcc to calculate the audio features
        mfccs = librosa.feature.mfcc(y=trimmed_sound, sr=sr)
        aver = np.average(mfccs)

        #store label and feature
        #the output should be a list
        #label and feature, corresponds one by one
        feature.append(aver)
        if filename[16:19] == 'cel':
            label.append('1')
        elif filename[16:19] == 'cla':
            label.append('2')
        elif filename[16:19] == 'flu':
            label.append('3')
        elif filename[16:19] == 'vio':
            label.append('4')

if __name__ == '__main__':
    main()
