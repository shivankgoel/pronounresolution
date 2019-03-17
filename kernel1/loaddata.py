from dependencies import *

#SUB_DATA_FOLDER = os.path.join(DATA_ROOT, 'gendered-pronoun-resolution')
#FAST_TEXT_DATA_FOLDER = os.path.join(DATA_ROOT, 'fasttext-crawl-300d-2m')

path = '../data/kaggle'
train_df = pd.read_csv(path+'/gap-development.tsv', sep='\t')
test_df = pd.read_csv(path+'/gap-test.tsv', sep='\t')
dev_df = pd.read_csv(path+'/gap-validation.tsv', sep='\t')


# sampletext = train_df['Text'][1]
# samplep = train_df['Pronoun-offset'][1]
# samplea = train_df['A-offset'][1]
# sampleb = train_df['B-offset'][1]

seq_list = list()

def extract_sents(text, char_offset_p, char_offset_a, char_offset_b, id):
	global seq_list
	seq_list.append(list())
	doc = nlp(text)
	token_idxs = [token.idx for token in doc]
	char_offsets = [char_offset_p, char_offset_a, char_offset_b]
	sent_list = list()
	for c in char_offsets:
		# char offset to token offset
		mention_offset = np.searchsorted(token_idxs,c)
		mention = doc[mention_offset]
		# token offset to sentence offset
		lens = [len([token for token in sent]) for sent in doc.sents]
		sumlens = list(np.cumsum(lens))
		sent_index = [i for i,j in enumerate(sumlens) if j>mention_offset][0]
		sent = list(doc.sents)[sent_index]
		# absolute position in the sentence
		sent_pos = mention_offset + 1
		if sent_index > 0:
			sent_pos = mention_offset - sumlens[sent_index-1]
		sent_list.append(sent.text)
		sent_list.append(sent_pos)
		seq_list[-1].append(sent)
	return pd.Series([id] + sent_list, index=['ID', 'Pronoun-Sent', 'Pronoun-Sent-Offset', 'A-Sent', 'A-Sent-Offset', 'B-Sent', 'B-Sent-Offset'])


def add_sent_columns(df):
	global seq_list
	seq_list = list()
	sent_df = df.apply(lambda row: extract_sents(row.loc['Text'], row['Pronoun-offset'], row['A-offset'], row['B-offset'], row['ID']), axis=1)
	df = df.join(sent_df.set_index('ID'), on='ID')
	return df, seq_list


train_df, train_tokenized = add_sent_columns(train_df)
test_df, test_tokenized = add_sent_columns(test_df)
dev_df, dev_tokenized = add_sent_columns(dev_df)

train_tokenized = train_tokenized[1:]
test_tokenized = test_tokenized[1:]
dev_tokenized = dev_tokenized[1:]


