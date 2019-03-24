from dependencies import *
from position_features import *
from embedding_features import *

num_embed_features = 11
embed_dim = 300


def create_embedding_features(df, text_column, offset_column):
    text_offset_list = df[[text_column, offset_column]].values.tolist()
    num_features = num_embed_features

    embed_feature_matrix = np.zeros(shape=(len(text_offset_list), num_features, embed_dim))
    for text_offset_index in range(len(text_offset_list)):
        text_offset = text_offset_list[text_offset_index]
        mention, parent, first_word, last_word, precedings2, followings2, precedings5, followings5, sent_tokens = embedding_features(
            text_offset[0], text_offset[1])

        feature_index = 0
        embed_feature_matrix[text_offset_index, feature_index, :] = mention.vector
        feature_index += 1
        embed_feature_matrix[text_offset_index, feature_index, :] = parent.vector
        feature_index += 1
        embed_feature_matrix[text_offset_index, feature_index, :] = first_word.vector
        feature_index += 1
        embed_feature_matrix[text_offset_index, feature_index, :] = last_word.vector
        feature_index += 1
        embed_feature_matrix[text_offset_index, feature_index:feature_index + 2, :] = np.asarray(
            [token.vector if token is not None else np.zeros((embed_dim,)) for token in precedings2])
        feature_index += len(precedings2)
        embed_feature_matrix[text_offset_index, feature_index:feature_index + 2, :] = np.asarray(
            [token.vector if token is not None else np.zeros((embed_dim,)) for token in followings2])
        feature_index += len(followings2)
        embed_feature_matrix[text_offset_index, feature_index, :] = np.mean(
            np.asarray([token.vector if token is not None else np.zeros((embed_dim,)) for token in precedings5]),
            axis=0)
        feature_index += 1
        embed_feature_matrix[text_offset_index, feature_index, :] = np.mean(
            np.asarray([token.vector if token is not None else np.zeros((embed_dim,)) for token in followings5]),
            axis=0)
        feature_index += 1
        embed_feature_matrix[text_offset_index, feature_index, :] = np.mean(
            np.asarray([token.vector for token in sent_tokens]), axis=0) if len(sent_tokens) > 0 else np.zeros(
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