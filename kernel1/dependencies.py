import numpy as np
import pandas as pd 

import os
import gc


from spacy.lang.en import English
from spacy.pipeline import DependencyParser
import spacy

#python -m spacy download en_core_web_lg
nlp = spacy.load("en_core_web_lg")


from nltk import Tree

from keras.preprocessing import sequence
from keras.preprocessing import text as ktext

from keras import backend
from keras import layers
from keras import models

from keras import initializers, regularizers, constraints, activations
from keras.engine import Layer
import keras.backend as K
from keras.layers import merge

import tensorflow as tf

from keras import callbacks as kc
from keras import optimizers as ko

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

import json

import pickle

result_save_path = './result/result_glove'
gender_dump_path = './data/gender_label_glove.pickle'
data_save_path = './data/data_gap_glove.pickle'


embedding_type = 'glove'  # {'None', 'glove', 'word2vec'}
word_embedding_path = 'glove.840B.300d.txt'

anonymous = 2  # {0: not using anonymous, 1: using anonymous only, 2: using anonymous and original data}

