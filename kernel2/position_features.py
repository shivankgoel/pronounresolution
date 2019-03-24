from spacy.lang.en import English
from spacy.pipeline import DependencyParser
import spacy
from nltk import Tree

nlp = spacy.load('en_core_web_lg')


def bs_(list_, target_):
    lo, hi = 0, len(list_) - 1

    while lo < hi:
        mid = lo + int((hi - lo) / 2)

        if target_ < list_[mid]:
            hi = mid
        elif target_ > list_[mid]:
            lo = mid + 1
        else:
            return mid
    return lo


def ohe_dist(dist, buckets):
    idx = bs_(buckets, dist)
    oh = np.zeros(shape=(len(buckets),), dtype=np.float32)
    oh[idx] = 1

    return oh

def position_features(text, char_offset1, char_offset2):
    doc = nlp(text)
    max_len = 64

    # char offset to token offset
    lens = [token.idx for token in doc]
    mention_offset1 = bs(lens, char_offset1) - 1
    mention_offset2 = bs(lens, char_offset2) - 1

    # token offset to sentence offset
    lens = [len(sent) for sent in doc.sents]
    acc_lens = [len_ for len_ in lens]
    pre_len = 0
    for i in range(0, len(acc_lens)):
        pre_len += acc_lens[i]
        acc_lens[i] = pre_len
    sent_index1 = bs(acc_lens, mention_offset1)
    sent_index2 = bs(acc_lens, mention_offset2)

    sent1 = list(doc.sents)[sent_index1]
    sent2 = list(doc.sents)[sent_index2]

    # buckets
    bucket_dist = [1, 2, 3, 4, 5, 8, 16, 32, 64]

    # relative distance
    dist = mention_offset2 - mention_offset1
    dist_oh = ohe_dist(dist, bucket_dist)

    # buckets
    bucket_pos = [0, 1, 2, 3, 4, 5, 8, 16, 32]

    # absolute position in the sentence
    sent_pos1 = mention_offset1 + 1
    if sent_index1 > 0:
        sent_pos1 = mention_offset1 - acc_lens[sent_index1 - 1]
    sent_pos_oh1 = ohe_dist(sent_pos1, bucket_pos)
    sent_pos_inv1 = len(sent1) - sent_pos1
    assert sent_pos_inv1 >= 0
    sent_pos_inv_oh1 = ohe_dist(sent_pos_inv1, bucket_pos)

    sent_pos2 = mention_offset2 + 1
    if sent_index2 > 0:
        sent_pos2 = mention_offset2 - acc_lens[sent_index2 - 1]
    sent_pos_oh2 = ohe_dist(sent_pos2, bucket_pos)
    sent_pos_inv2 = len(sent2) - sent_pos2
    if sent_pos_inv2 < 0:
        print(sent_pos_inv2)
        print(len(sent2))
        print(sent_pos2)
        raise ValueError
    sent_pos_inv_oh2 = ohe_dist(sent_pos_inv2, bucket_pos)

    sent_pos_ratio1 = sent_pos1 / len(sent1)
    sent_pos_ratio2 = sent_pos2 / len(sent2)

    return dist_oh, sent_pos_oh1, sent_pos_oh2, sent_pos_inv_oh1, sent_pos_inv_oh2

