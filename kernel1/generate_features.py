from dependencies import *
from position_features import *
from embedding_features import *

num_pos_features = 45


def create_dist_features(df, text_column, pronoun_offset_column, name_offset_column):
    text_offset_list = df[[text_column, pronoun_offset_column, name_offset_column]].values.tolist()
    num_features = num_pos_features

    pos_feature_matrix = np.zeros(shape=(len(text_offset_list), num_features))
    for text_offset_index in range(len(text_offset_list)):
        text_offset = text_offset_list[text_offset_index]
        dist_oh, sent_pos_oh1, sent_pos_oh2, sent_pos_inv_oh1, sent_pos_inv_oh2 = extrac_positional_features(
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
