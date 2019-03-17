from dependencies import *

from b import *

num_pos_features = 45
max_len = 50


def ohe_dist(dist, buckets):
    idx = min(np.searchsorted(buckets,dist),len(buckets)-1)   
    oh = np.zeros(shape=(len(buckets),), dtype=np.float32)
    oh[idx] = 1
    return oh


def extrac_positional_features(text, char_offset1, char_offset2):
    #text = "Zoe Telford -- played the police officer girlfriend of Simon, Maggie. Dumped by Simon in the final episode of series 1, after he slept with Jenny, and is not seen again. Phoebe Thomas played Cheryl Cassidy, Pauline's friend and also a year 11 pupil in Simon's class. Dumped her boyfriend following Simon's advice after he wouldn't have sex with her but later realised this was due to him catching crabs off her friend Pauline."
    #char_offset1 = 274
    #char_offset2 = 191
    doc = nlp(text)
    max_len = 64
    # char offset to token offset
    token_idxs = [token.idx for token in doc]
    mention_offset1 = np.searchsorted(token_idxs,char_offset1)
    mention_offset2 = np.searchsorted(token_idxs,char_offset2)
    # token offset to sentence offset
    lens = [len([token for token in sent]) for sent in doc.sents]
    sumlens = list(np.cumsum(lens))
    sent_index1 = [i for i,j in enumerate(sumlens) if j>mention_offset1][0]
    sent_index2 = [i for i,j in enumerate(sumlens) if j>mention_offset2][0]
    #
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
        sent_pos1 = mention_offset1 - sumlens[sent_index1-1]
    sent_pos_oh1 = ohe_dist(sent_pos1, bucket_pos)
    sent_pos_inv1 = len(sent1) - sent_pos1
    sent_pos_inv_oh1 = ohe_dist(sent_pos_inv1, bucket_pos)
    #
    sent_pos2 = mention_offset2 + 1
    if sent_index2 > 0:
        sent_pos2 = mention_offset2 - sumlens[sent_index2-1]
    sent_pos_oh2 = ohe_dist(sent_pos2, bucket_pos)
    sent_pos_inv2 = len(sent2) - sent_pos2
    sent_pos_inv_oh2 = ohe_dist(sent_pos_inv2, bucket_pos)
    #
    return dist_oh, sent_pos_oh1, sent_pos_oh2, sent_pos_inv_oh1, sent_pos_inv_oh2


def create_dist_features(df, text_column, pronoun_offset_column, name_offset_column):
    text_offset_list = df[[text_column, pronoun_offset_column, name_offset_column]].values.tolist()
    num_features = num_pos_features
    pos_feature_matrix = np.zeros(shape=(len(text_offset_list), num_features))
    for text_offset_index in range(len(text_offset_list)):
        text_offset = text_offset_list[text_offset_index]
        dist_oh, sent_pos_oh1, sent_pos_oh2, sent_pos_inv_oh1, sent_pos_inv_oh2 = extrac_positional_features(text_offset[0], text_offset[1], text_offset[2])
        feature_index = 0
        pos_feature_matrix[text_offset_index, feature_index:feature_index+len(dist_oh)] = np.asarray(dist_oh)
        feature_index += len(dist_oh)
        pos_feature_matrix[text_offset_index, feature_index:feature_index+len(sent_pos_oh1)] = np.asarray(sent_pos_oh1)
        feature_index += len(sent_pos_oh1)
        pos_feature_matrix[text_offset_index, feature_index:feature_index+len(sent_pos_oh2)] = np.asarray(sent_pos_oh2)
        feature_index += len(sent_pos_oh2)
        pos_feature_matrix[text_offset_index, feature_index:feature_index+len(sent_pos_inv_oh1)] = np.asarray(sent_pos_inv_oh1)
        feature_index += len(sent_pos_inv_oh1)
        pos_feature_matrix[text_offset_index, feature_index:feature_index+len(sent_pos_inv_oh2)] = np.asarray(sent_pos_inv_oh2)
        feature_index += len(sent_pos_inv_oh2)
    return pos_feature_matrix


pa_pos_tra = create_dist_features(train_df, 'Text', 'Pronoun-offset', 'A-offset')
pa_pos_dev = create_dist_features(dev_df, 'Text', 'Pronoun-offset', 'A-offset')
pa_pos_test = create_dist_features(test_df, 'Text', 'Pronoun-offset', 'A-offset')

pb_pos_tra = create_dist_features(train_df, 'Text', 'Pronoun-offset', 'B-offset')
pb_pos_dev = create_dist_features(dev_df, 'Text', 'Pronoun-offset', 'B-offset')
pb_pos_test = create_dist_features(test_df, 'Text', 'Pronoun-offset', 'B-offset')


X_train = [train_p_seq, train_a_seq, train_b_seq , train_p_pos, train_a_pos, train_b_pos , index_p_train, index_a_train, index_b_train, pa_pos_tra, pb_pos_tra]
X_dev = [dev_p_seq, dev_a_seq, dev_b_seq , dev_p_pos, dev_a_pos, dev_b_pos , index_p_dev, index_a_dev, index_b_dev, pa_pos_dev, pb_pos_dev]
X_test = [test_p_seq, test_a_seq, test_b_seq, test_p_pos, test_a_pos, test_b_pos, index_p_test, index_a_test, index_b_test, pa_pos_test, pb_pos_test]


