# Music Instrument Classifier

This is a simple classifier that is able to detect single-note sounds of various musical instruments.

Currently supported types are cello, clarinet, flute and violin. Piano will be supported soon. 


## Dataset

All the data used are collected from London Philharmonic Orchestra Dataset (http://www.philharmonia.co.uk/explore/sound_samples). Original audio files not included in this repository. 


## Dependencies
 - numpy
 - scikit-learn (sklearn)
 - librosa
 - glob


## Algorithm

### Preprocessing

The music pieces have their leading and ending silence trimmed.

### Feature Extraction

The Mel Frequency Cepstral Coefficents (MFCCs) of each music piece is extracted. For each audio file, its MFCCs are averaged to produce the final, length-20 feature vector. 

### Classification

An SVM classifier is trained from the feature vectors to determine the instrument it belongs to. 


## Demo (TODO)
