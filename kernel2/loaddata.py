from dependencies import *

from generate_features import *
from position_features import *

path = '../data/kaggle'
train_df = pd.read_csv(path+'/gap-development.tsv', sep='\t')
test_df = pd.read_csv(path+'/gap-test.tsv', sep='\t')
dev_df = pd.read_csv(path+'/gap-validation.tsv', sep='\t')


M = {'him', 'he', 'his'}
F = {'her', 'she', 'hers'}

train_gender = list()
test_gender = list()
dev_gender = list()

for pronoun in train_df['Pronoun']:
    if pronoun.lower() in M:
        train_gender.append(0)
    elif pronoun.lower() in F:
        train_gender.append(1)
    else:
        print('error')

for pronoun in test_df['Pronoun']:
    if pronoun.lower() in M:
        test_gender.append(0)
    elif pronoun.lower() in F:
        test_gender.append(1)
    else:
        print('error')

for pronoun in dev_df['Pronoun']:
    if pronoun.lower() in M:
        dev_gender.append(0)
    elif pronoun.lower() in F:
        dev_gender.append(1)
    else:
        print('error')

gender = dict()
gender['train'] = train_gender
gender['test'] = test_gender
gender['dev'] = dev_gender

with open(gender_dump_path, 'wb') as f:
    pickle.dump(gender, f, protocol=pickle.HIGHEST_PROTOCOL)
f.close()



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


data = dict()
data['p_emb_tra'] = p_emb_tra
data['p_emb_dev'] = p_emb_dev
data['p_emb_test'] = p_emb_test

data['a_emb_tra'] = a_emb_tra
data['a_emb_dev'] = a_emb_dev
data['a_emb_test'] = a_emb_test

data['b_emb_tra'] = b_emb_tra
data['b_emb_dev'] = b_emb_dev
data['b_emb_test'] = b_emb_test

data['pa_pos_tra'] = pa_pos_tra
data['pa_pos_dev'] = pa_pos_dev
data['pa_pos_test'] = pa_pos_test

data['pb_pos_tra'] = pb_pos_tra
data['pb_pos_dev'] = pb_pos_dev
data['pb_pos_test'] = pb_pos_test

data['y_tra'] = y_tra
data['y_dev'] = y_dev
data['y_test'] = y_test

with open(data_save_path, 'wb') as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
f.close()