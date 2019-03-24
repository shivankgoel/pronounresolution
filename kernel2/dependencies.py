from spacy.lang.en import English
from spacy.pipeline import DependencyParser
import spacy
from nltk import Tree


from embedding_features import embedding_features
from position_features import position_features
import numpy as np

import pandas as pd

nlp = spacy.load('en_core_web_lg')