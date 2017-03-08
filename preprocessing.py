import numpy as np
import sklearn
import librosa
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

def feature_extract():

    sr = 44100
    window_size = 2048
    hop_size = window_size/2
    data = []

    #read file
    files = glob.glob('final_data/data/*/*.mp3')
    np.random.shuffle(files)
    for filename in files:

        music, sr= librosa.load(filename, sr = sr)

        start_trim = detect_leading_silence(music)
        end_trim = detect_leading_silence(np.flipud(music))

        duration = len(music)
        trimmed_sound = music[start_trim:duration-end_trim]
        # the sound without silence

        #use mfcc to calculate the audio features
        mfccs = librosa.feature.mfcc(y=trimmed_sound, sr=sr)
        aver = np.mean(mfccs, axis = 1)
        feature = aver.reshape(20)

        #store label and feature
        #the output should be a list
        #label and feature, corresponds one by one
        #feature.append(aver)
        if filename[16:19] == 'cel':
            label = 1
        elif filename[16:19] == 'cla':
            label = 2
        elif filename[16:19] == 'flu':
            label = 3
        elif filename[16:19] == 'vio':
            label = 4

        data2 = [filename, feature, label]
        # print data2
        # print feature.shape
        data.append(data2)
        #data = np.vstack((data, data2))
        # print data
    return data

def main():
    data = feature_extract()
    print data
    print len(data)

# if __name__ == '__main__':
#     main()
