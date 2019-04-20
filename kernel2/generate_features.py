from dependencies import *
from position_features import *
from embedding_features import *

num_embed_features = 11
embed_dim = 300


df = pd.read_csv('/Users/shivankgoel/Desktop/Projects/Projects/ALT/glove.840B.300d.txt', sep=" ", quoting=3, header=None, index_col=0)
glovemodel = {key: val.values for key, val in df.T.items()}

from gensim.models import KeyedVectors
#modeldebiased = KeyedVectors.load_word2vec_format('/Users/shivankgoel/Desktop/Projects/Projects/ALT/GoogleNews-vectors-negative300-hard-debiased.bin', binary=True)
#modelsimple = KeyedVectors.load_word2vec_format('/Users/shivankgoel/Desktop/Projects/Projects/ALT/GoogleNews-vectors-negative300.bin', binary=True)

model = glovemodel

def give_my_embedding(word,gendered=True,embed_dim=300):
    if word is None:
        return np.zeros((embed_dim,))
    elif not gendered:
        return word.vector
    elif word.text in model:
        return np.float32(model[word.text])
    else:
        return np.float32(np.zeros((embed_dim,)))


#list(give_my_embedding(nlp(word),False)) == list(np.float32(give_my_embedding('cat')))


def create_embedding_features(df, text_column, offset_column):
    text_offset_list = df[[text_column, offset_column]].values.tolist()
    num_features = num_embed_features

    embed_feature_matrix = np.zeros(shape=(len(text_offset_list), num_features, embed_dim))
    for text_offset_index in range(len(text_offset_list)):
        text_offset = text_offset_list[text_offset_index]
        mention, parent, first_word, last_word, precedings2, followings2, precedings5, followings5, sent_tokens = embedding_features(
            text_offset[0], text_offset[1])

        feature_index = 0
        embed_feature_matrix[text_offset_index, feature_index, :] = give_my_embedding(mention)
        feature_index += 1
        embed_feature_matrix[text_offset_index, feature_index, :] = give_my_embedding(parent)
        feature_index += 1
        embed_feature_matrix[text_offset_index, feature_index, :] = give_my_embedding(first_word)
        feature_index += 1
        embed_feature_matrix[text_offset_index, feature_index, :] = give_my_embedding(last_word)
        feature_index += 1
        embed_feature_matrix[text_offset_index, feature_index:feature_index + 2, :] = np.asarray(
            [give_my_embedding(token) for token in precedings2])
        feature_index += len(precedings2)
        embed_feature_matrix[text_offset_index, feature_index:feature_index + 2, :] = np.asarray(
            [give_my_embedding(token) for token in followings2])
        feature_index += len(followings2)
        embed_feature_matrix[text_offset_index, feature_index, :] = np.mean(
            np.asarray([give_my_embedding(token) for token in precedings5]),
            axis=0)
        feature_index += 1
        embed_feature_matrix[text_offset_index, feature_index, :] = np.mean(
            np.asarray([give_my_embedding(token) for token in followings5]),
            axis=0)
        feature_index += 1
        embed_feature_matrix[text_offset_index, feature_index, :] = np.mean(
            np.asarray([give_my_embedding(token) for token in sent_tokens]), axis=0) if len(sent_tokens) > 0 else np.zeros(
            embed_dim)
        feature_index += 1

    return embed_feature_matrix


num_pos_features = 45


def create_dist_features(df, text_column, pronoun_offset_column, name_offset_column):
    text_offset_list = df[[text_column, pronoun_offset_column, name_offset_column]].values.tolist()
    num_features = num_pos_features

    pos_feature_matrix = np.zeros(shape=(len(text_offset_list), num_features))
    for text_offset_index in range(len(text_offset_list)):
        text_offset = text_offset_list[text_offset_index]
        dist_oh, sent_pos_oh1, sent_pos_oh2, sent_pos_inv_oh1, sent_pos_inv_oh2 = position_features(
            text_offset[0], text_offset[1], text_offset[2])

        feature_index = 0
        pos_feature_matrix[text_offset_index, feature_index:feature_index + len(dist_oh)] = np.asarray(dist_oh)
        feature_index += len(dist_oh)
        pos_feature_matrix[text_offset_index, feature_index:feature_index + len(sent_pos_oh1)] = np.asarray(
            sent_pos_oh1)
        feature_index += len(sent_pos_oh1)
        pos_feature_matrix[text_offset_index, feature_index:feature_index + len(sent_pos_oh2)] = np.asarray(
            sent_pos_oh2)
        feature_index += len(sent_pos_oh2)
        pos_feature_matrix[text_offset_index, feature_index:feature_index + len(sent_pos_inv_oh1)] = np.asarray(
            sent_pos_inv_oh1)
        feature_index += len(sent_pos_inv_oh1)
        pos_feature_matrix[text_offset_index, feature_index:feature_index + len(sent_pos_inv_oh2)] = np.asarray(
            sent_pos_inv_oh2)
        feature_index += len(sent_pos_inv_oh2)

    return pos_feature_matrix