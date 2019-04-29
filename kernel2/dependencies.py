from spacy.lang.en import English
from spacy.pipeline import DependencyParser
import spacy
from nltk import Tree

import os
import gc

import numpy as np

import pandas as pd

nlp = spacy.load('en_core_web_lg')


import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import gensim

import pickle

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

result_save_path = './result/result_glove_kaggle_kaggle'
gender_dump_path = './data/gender_label_glove_kaggle_kaggle.pickle'
data_save_path = './data/data_gap_glove_kaggle_kaggle.pickle'