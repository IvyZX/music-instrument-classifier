import numpy as np
import sklearn
import librosa
from sklearn import svm
from sklearn.externals import joblib
from sklearn import decomposition
from sklearn import neighbors
import preprocessing
import classify

def feature_extract(audio_filename):

    sr = 44100
    window_size = 2048
    hop_size = window_size/2
    print audio_filename
    if audio_filename[5] == '1':
        print "This instrument is cello."
    elif audio_filename[5] == '2':
        print "This instrument is clarinet."
    elif audio_filename[5] == '3':
        print "This instrument is flut."
    elif audio_filename[5] == '4':
        print "This instrument is violin."
    elif audio_filename[5] == '5':
        print "This instrument is piano."

    music, sr= librosa.load(audio_filename, sr = sr)
    start_trim = preprocessing.detect_leading_silence(music)
    end_trim = preprocessing.detect_leading_silence(np.flipud(music))

    duration = len(music)
    trimmed_sound = music[start_trim:duration-end_trim]

    mfccs = librosa.feature.mfcc(y=trimmed_sound, sr=sr)
    aver = np.mean(mfccs, axis = 1)
    audio_feature = aver.reshape(20)
    return audio_feature


def result(pre):
    if pre == 1:
        print "The prediction of this instrument is cello."
    elif pre == 2:
        print "The prediction of this instrument is clarinet."
    elif pre == 3:
        print "The prediction of this instrument is flut."
    elif pre == 4:
        print "The prediction of this instrument is violin."
    elif pre == 5:
        print "The prediction of this instrument is piano."

def main():
    audio_filename = input("Please choose an audio file(test/1-10.mp3): ")
    demo_data = feature_extract(audio_filename)
    print 'processed data.'
    model_params = {
        'pca_n': 10,
        'knn_k': 5,
        'knn_metric': 'minkowski'
    }
    #  train_and_test(data, [model_params, 'svc'])
    model = classify.load_model(model_params)
    pre = classify.predict(model, demo_data, [model_params, 'svc'])
    result(pre)

if __name__ == '__main__':
    main()
