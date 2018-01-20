# -*- coding: utf-8 -*-

import os
import pandas as pd

#path = r'F:\Juhon\python\EdX Azure\Python for DS\Module2\Datasets'
#file = 'direct_marketing.csv'
#
#df = pd.read_csv(os.path.join(path, file))
#
#df[ (df.recency < 7) & (df.newbie == 0) ]

### ordinal categories
#ordered_satisfaction = ['Very Unhappy', 'Unhappy', 'Neutral', 'Happy', 'Very Happy']
#df = pd.DataFrame({'satisfaction':['Mad', 'Happy', 'Unhappy', 'Neutral']})
#df.satisfaction = df.satisfaction.astype("category",
#  ordered=True,
#  categories=ordered_satisfaction
#).cat.codes

### bag of words
#from sklearn.feature_extraction.text import CountVectorizer
#
#corpus = [
#"Authman ran faster than Harry because he is an athlete.",
#"Authman and Harry ran faster and faster.",
#]
#bow = CountVectorizer()
#X = bow.fit_transform(corpus) # Sparse Matrix
#bow.get_feature_names()
#X.toarray()

### basic image featurization
## Uses the Image module (PIL)
#from scipy import misc
#img = misc.imread('image.png') # Load the image up
#img = img[::2, ::2] # resample down (if needed)
## Scale colors from (0-255) to (0-1), then reshape to 1D array per pixel, e.g. grayscale
## If you had color images and wanted to preserve all color channels, use .reshape(-1,3)
#X = (img / 255.0).reshape(-1)
## To-Do: Machine Learning with X!

### set of images
#dset = []
#for fname in files:
#  img = misc.imread(fname)
#  dset.append(  (img[::2, ::2] / 255.0).reshape(-1)  )
#dset = pd.DataFrame( dset )

### audio
#import scipy.io.wavfile as wavfile
#sample_rate, audio_data = wavfile.read('sound.wav')
#print audio_data
## To-Do: Machine Learning with audio_data!

### fillna methods
#df.my_feature.fillna( df.my_feature.mean() )
#df.fillna(0)
## be mindful of axis
#df.fillna(method='ffill')  # fill the values forward
#df.fillna(method='bfill')  # fill the values in reverse
#df.fillna(limit=5)
#df.interpolate(method='polynomial', order=2)

#df = df.dropna(axis=0, thresh=2).drop(labels=['ColA', axis=1]).drop_duplicates(subset=['ColB', 'ColC']).reset_index()
