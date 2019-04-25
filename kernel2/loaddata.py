from dependencies import *

from generate_features import *
from position_features import *

path = '../data/kaggle'
train_df = pd.read_csv(path+'/gap-development.tsv', sep='\t')
test_df = pd.read_csv(path+'/gap-test.tsv', sep='\t')
dev_df = pd.read_csv(path+'/gap-validation.tsv', sep='\t')

p_emb_tra = create_embedding_features(train_df, 'Text', 'Pronoun-offset')
p_emb_dev = create_embedding_features(dev_df, 'Text', 'Pronoun-offset')
p_emb_test = create_embedding_features(test_df, 'Text', 'Pronoun-offset')


a_emb_tra = create_embedding_features(train_df, 'Text', 'A-offset')
a_emb_dev = create_embedding_features(dev_df, 'Text', 'A-offset')
a_emb_test = create_embedding_features(test_df, 'Text', 'A-offset')

b_emb_tra = create_embedding_features(train_df, 'Text', 'B-offset')
b_emb_dev = create_embedding_features(dev_df, 'Text', 'B-offset')
b_emb_test = create_embedding_features(test_df, 'Text', 'B-offset')

'''
p_emb_tra[0].shape = (11,300)
len(p_emb_tra) = 2000
len(p_emb_dev) = 454
len(p_emb_test) = 2000
'''

pa_pos_tra = create_dist_features(train_df, 'Text', 'Pronoun-offset', 'A-offset')
pa_pos_dev = create_dist_features(dev_df, 'Text', 'Pronoun-offset', 'A-offset')
pa_pos_test = create_dist_features(test_df, 'Text', 'Pronoun-offset', 'A-offset')

pb_pos_tra = create_dist_features(train_df, 'Text', 'Pronoun-offset', 'B-offset')
pb_pos_dev = create_dist_features(dev_df, 'Text', 'Pronoun-offset', 'B-offset')
pb_pos_test = create_dist_features(test_df, 'Text', 'Pronoun-offset', 'B-offset')

'''
len(pa_pos_tra) = 2000
pa_pos_tra[0].shape = (45,)
We have 45 position features.
'''


# p_emb_dev = create_embedding_features(dev_df, 'Text', 'Pronoun-offset')
# a_emb_dev = create_embedding_features(dev_df, 'Text', 'A-offset')
# b_emb_dev = create_embedding_features(dev_df, 'Text', 'B-offset')
# pa_pos_dev = create_dist_features(dev_df, 'Text', 'Pronoun-offset', 'A-offset')
# pb_pos_dev = create_dist_features(dev_df, 'Text', 'Pronoun-offset', 'B-offset')



def _row_to_y(row):
    if row.loc['A-coref']:
        return 0
    if row.loc['B-coref']:
        return 1
    return 2

y_tra = train_df.apply(_row_to_y, axis=1)
y_dev = dev_df.apply(_row_to_y, axis=1)
y_test = test_df.apply(_row_to_y, axis=1)