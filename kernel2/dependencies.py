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