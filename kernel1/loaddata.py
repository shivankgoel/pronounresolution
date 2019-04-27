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

with open('gender_label.pickle', 'wb') as f:
    pickle.dump(gender, f, protocol=pickle.HIGHEST_PROTOCOL)
f.close()


seq_list = list()
train_df, train_tokenized = add_sent_columns(train_df, 'Text', 'Pronoun-offset', 'A-offset', 'B-offset')
seq_list = list()
test_df, test_tokenized = add_sent_columns(test_df, 'Text', 'Pronoun-offset', 'A-offset', 'B-offset')
seq_list = list()
dev_df, dev_tokenized = add_sent_columns(dev_df, 'Text', 'Pronoun-offset', 'A-offset', 'B-offset')

# df apply will call the first row twice, remove the first one
train_tokenized = train_tokenized[1:]
test_tokenized = test_tokenized[1:]
dev_tokenized = dev_tokenized[1:]


embed_size = 300
max_features = 80000


# generate word index
word_index = dict()
idx = 1
for text_ in train_tokenized + test_tokenized + dev_tokenized:
    for sent_ in text_:
        for word_ in sent_:
            if word_.text not in word_index and nlp.vocab.has_vector(word_.text):
                word_index[word_.text] = idx
                idx += 1

nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words + 1, embed_size))

for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = None
    if nlp.vocab.has_vector(word):
        embedding_vector = nlp.vocab.vectors[nlp.vocab.strings[word]]
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

print(embedding_matrix.shape)

# generate pos tag index
pos_index = dict()
idx = 1
for text_ in train_tokenized + test_tokenized + dev_tokenized:
    for sent_ in text_:
        for word_ in sent_:
            if word_.pos not in pos_index:
                pos_index[word_.pos] = idx
                idx += 1

for text_ in test_tokenized:
    for sent_ in text_:
        for word_ in sent_:
            if word_.pos not in pos_index:
                pos_index[word_.pos] = idx
                idx += 1

print(len(pos_index))

train_p_tokenized = sentences_to_sequences([row[0] for row in train_tokenized], word_index)
train_a_tokenized = sentences_to_sequences([row[1] for row in train_tokenized], word_index)
train_b_tokenized = sentences_to_sequences([row[2] for row in train_tokenized], word_index)

test_p_tokenized = sentences_to_sequences([row[0] for row in test_tokenized], word_index)
test_a_tokenized = sentences_to_sequences([row[1] for row in test_tokenized], word_index)
test_b_tokenized = sentences_to_sequences([row[2] for row in test_tokenized], word_index)

dev_p_tokenized = sentences_to_sequences([row[0] for row in dev_tokenized], word_index)
dev_a_tokenized = sentences_to_sequences([row[1] for row in dev_tokenized], word_index)
dev_b_tokenized = sentences_to_sequences([row[2] for row in dev_tokenized], word_index)

seq_p_train = sequence.pad_sequences(train_p_tokenized, maxlen = max_len, padding='post')
seq_a_train = sequence.pad_sequences(train_a_tokenized, maxlen = max_len, padding='post')
seq_b_train = sequence.pad_sequences(train_b_tokenized, maxlen = max_len, padding='post')

seq_p_test = sequence.pad_sequences(test_p_tokenized, maxlen = max_len, padding='post')
seq_a_test = sequence.pad_sequences(test_a_tokenized, maxlen = max_len, padding='post')
seq_b_test = sequence.pad_sequences(test_b_tokenized, maxlen = max_len, padding='post')

seq_p_dev = sequence.pad_sequences(dev_p_tokenized, maxlen = max_len, padding='post')
seq_a_dev = sequence.pad_sequences(dev_a_tokenized, maxlen = max_len, padding='post')
seq_b_dev = sequence.pad_sequences(dev_b_tokenized, maxlen = max_len, padding='post')


train_p_pos = poses_to_sequences([row[0] for row in train_tokenized], pos_index)
train_a_pos = poses_to_sequences([row[1] for row in train_tokenized], pos_index)
train_b_pos = poses_to_sequences([row[2] for row in train_tokenized], pos_index)

test_p_pos = poses_to_sequences([row[0] for row in test_tokenized], pos_index)
test_a_pos = poses_to_sequences([row[1] for row in test_tokenized], pos_index)
test_b_pos = poses_to_sequences([row[2] for row in test_tokenized], pos_index)

dev_p_pos = poses_to_sequences([row[0] for row in dev_tokenized], pos_index)
dev_a_pos = poses_to_sequences([row[1] for row in dev_tokenized], pos_index)
dev_b_pos = poses_to_sequences([row[2] for row in dev_tokenized], pos_index)

pos_p_train = sequence.pad_sequences(train_p_pos, maxlen = max_len, padding='post')
pos_a_train = sequence.pad_sequences(train_a_pos, maxlen = max_len, padding='post')
pos_b_train = sequence.pad_sequences(train_b_pos, maxlen = max_len, padding='post')

pos_p_test = sequence.pad_sequences(test_p_pos, maxlen = max_len, padding='post')
pos_a_test = sequence.pad_sequences(test_a_pos, maxlen = max_len, padding='post')
pos_b_test = sequence.pad_sequences(test_b_pos, maxlen = max_len, padding='post')

pos_p_dev = sequence.pad_sequences(dev_p_pos, maxlen = max_len, padding='post')
pos_a_dev = sequence.pad_sequences(dev_a_pos, maxlen = max_len, padding='post')
pos_b_dev = sequence.pad_sequences(dev_b_pos, maxlen = max_len, padding='post')

index_p_train = train_df['Pronoun-Sent-Offset'].values
index_a_train = train_df['A-Sent-Offset'].values
index_b_train = train_df['B-Sent-Offset'].values

index_p_test = test_df['Pronoun-Sent-Offset'].values
index_a_test = test_df['A-Sent-Offset'].values
index_b_test = test_df['B-Sent-Offset'].values

index_p_dev = dev_df['Pronoun-Sent-Offset'].values
index_a_dev = dev_df['A-Sent-Offset'].values
index_b_dev = dev_df['B-Sent-Offset'].values

pa_pos_tra = create_dist_features(train_df, 'Text', 'Pronoun-offset', 'A-offset')
pa_pos_dev = create_dist_features(dev_df, 'Text', 'Pronoun-offset', 'A-offset')
pa_pos_test = create_dist_features(test_df, 'Text', 'Pronoun-offset', 'A-offset')

pb_pos_tra = create_dist_features(train_df, 'Text', 'Pronoun-offset', 'B-offset')
pb_pos_dev = create_dist_features(dev_df, 'Text', 'Pronoun-offset', 'B-offset')
pb_pos_test = create_dist_features(test_df, 'Text', 'Pronoun-offset', 'B-offset')

X_train = [seq_p_train, seq_a_train, seq_b_train, pos_p_train, pos_a_train, pos_b_train, index_p_train, index_a_train, index_b_train, pa_pos_tra, pb_pos_tra]
X_dev = [seq_p_dev, seq_a_dev, seq_b_dev, pos_p_dev, pos_a_dev, pos_b_dev, index_p_dev, index_a_dev, index_b_dev, pa_pos_dev, pb_pos_dev]
X_test = [seq_p_test, seq_a_test, seq_b_test, pos_p_test, pos_a_test, pos_b_test, index_p_test, index_a_test, index_b_test, pa_pos_test, pb_pos_test]


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
data['X_train'] = X_train
data['X_dev'] = X_dev
data['X_test'] = X_test

data['y_tra'] = y_tra
data['y_dev'] = y_dev
data['y_test'] = y_test

data['embedding'] = embedding_matrix

data['position'] = pos_index

with open('data_gap.pickle', 'wb') as f:
   pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
f.close()
