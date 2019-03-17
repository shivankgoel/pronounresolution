from dependencies import *

from loaddata import *

embed_size = 300
max_features = 80000

# generate word index
def give_word2idx():
	word2idx = dict()
	pos2idx = dict()
	idx = 1
	idx2 = 1
	for lines in train_tokenized+test_tokenized+dev_tokenized:
		for line in lines:
			for word in line:
				if word.pos not in pos2idx:
					pos2idx[word.pos] = idx2
					idx2+=1
				if word.text not in word2idx and nlp.vocab.has_vector(word.text):
					word2idx[word.text] = idx
					idx += 1
	return word2idx,pos2idx

word2idx, pos2idx = give_word2idx()

#You can limit vocab here *******
vocab_size = len(word2idx)
embedding_matrix = np.zeros((vocab_size + 1, embed_size))
		
for word, i in word2idx.items():
	embedding_matrix[i] = nlp.vocab.vectors[nlp.vocab.strings[word]]
		
print(embedding_matrix.shape)

def change_sent_2_wordidx(sent):
	return [word2idx[token.text] if token.text in word2idx else 0 for token in sent]

def change_sent_2_posidx(sent):
	return [pos2idx[token.pos] if token.pos in pos2idx else 0 for token in sent]


max_len = 50
def convert_to_wordidx(inp_tokenized):
	p_tokenized = sequence.pad_sequences([change_sent_2_wordidx(row[0]) for row in inp_tokenized],maxlen = max_len, padding='post')
	a_tokenized = sequence.pad_sequences([change_sent_2_wordidx(row[0]) for row in inp_tokenized],maxlen = max_len, padding='post')
	b_tokenized = sequence.pad_sequences([change_sent_2_wordidx(row[0]) for row in inp_tokenized],maxlen = max_len, padding='post')
	return p_tokenized,a_tokenized,b_tokenized


def convert_to_posidx(inp_tokenized):
	p_tokenized = sequence.pad_sequences([change_sent_2_posidx(row[0]) for row in inp_tokenized],maxlen = max_len, padding='post')
	a_tokenized = sequence.pad_sequences([change_sent_2_posidx(row[0]) for row in inp_tokenized],maxlen = max_len, padding='post')
	b_tokenized = sequence.pad_sequences([change_sent_2_posidx(row[0]) for row in inp_tokenized],maxlen = max_len, padding='post')
	return p_tokenized,a_tokenized,b_tokenized

def give_offset_values(data_df):
	return data_df['Pronoun-Sent-Offset'].values, data_df['A-Sent-Offset'].values, data_df['B-Sent-Offset'].values


train_p_seq, train_a_seq, train_b_seq  = convert_to_wordidx(train_tokenized)
test_p_seq, test_a_seq, test_b_seq  = convert_to_wordidx(test_tokenized)
dev_p_seq, dev_a_seq, dev_b_seq  = convert_to_wordidx(dev_tokenized)

train_p_pos, train_a_pos, train_b_pos  = convert_to_posidx(train_tokenized)
test_p_pos, test_a_pos, test_b_pos  = convert_to_posidx(test_tokenized)
dev_p_pos, dev_a_pos, dev_b_pos  = convert_to_posidx(dev_tokenized)

index_p_train , index_a_train , index_b_train = give_offset_values(train_df)
index_p_test , index_a_test , index_b_test = give_offset_values(test_df)
index_p_dev , index_a_dev , index_b_dev = give_offset_values(dev_df)

def _row_to_y(row):
    if row.loc['A-coref']:
        return 0
    if row.loc['B-coref']:
        return 1
    return 2

y_tra = train_df.apply(_row_to_y, axis=1)
y_dev = dev_df.apply(_row_to_y, axis=1)
y_test = test_df.apply(_row_to_y, axis=1)


