import numpy as np
import pandas as pd 

import os
import gc


from spacy.lang.en import English
from spacy.pipeline import DependencyParser
import spacy

#python -m spacy download en_core_web_lg
nlp = spacy.load("en_core_web_lg")


import seaborn as sns

from nltk import Tree

from keras.preprocessing import sequence
from keras.preprocessing import text as ktext