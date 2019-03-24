from spacy.lang.en import English
from spacy.pipeline import DependencyParser
import spacy
from nltk import Tree

import os
import gc

import numpy as np

import pandas as pd

nlp = spacy.load('en_core_web_lg')