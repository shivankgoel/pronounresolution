from dependencies import *
from position_features import bs

max_len = 50 # longer than 99% of the sentences

seq_list = list()


def extract_sents(text, char_offset_p, char_offset_a, char_offset_b, id):
    global max_len
    global seq_list

    seq_list.append(list())

    doc = nlp(text)
    token_lens = [token.idx for token in doc]

    char_offsets = [char_offset_p, char_offset_a, char_offset_b]
    sent_list = list()

    for char_offset in char_offsets:
        # char offset to token offset
        mention_offset = bs(token_lens, char_offset) - 1
        # mention_word
        mention = doc[mention_offset]

        # token offset to sentence offset
        lens = [len(sent) for sent in doc.sents]
        acc_lens = [len_ for len_ in lens]
        pre_len = 0
        for i in range(0, len(acc_lens)):
            pre_len += acc_lens[i]
            acc_lens[i] = pre_len
        sent_index = bs(acc_lens, mention_offset)
        # mention sentence
        sent = list(doc.sents)[sent_index]

        # absolute position in the sentence
        sent_pos = mention_offset + 1
        if sent_index > 0:
            sent_pos = mention_offset - acc_lens[sent_index - 1]

        # clip the sentence if it is longer than max length
        if len(sent) > max_len:
            # make sure the mention is in the sentence span
            if sent_pos < max_len - 1:
                sent_list.append(sent[0:max_len].text)
                sent_list.append(sent_pos)
                seq_list[-1].append(sent[0:max_len])
            else:
                sent_list.append(sent[sent_pos - max_len + 2: min(sent_pos + 2, len(sent))].text)
                sent_list.append(max_len - 2)
                seq_list[-1].append(sent[sent_pos - max_len + 2: min(sent_pos + 2, len(sent))])
        else:
            sent_list.append(sent.text)
            sent_list.append(sent_pos)
            seq_list[-1].append(sent)

    return pd.Series([id] + sent_list,
                     index=['ID', 'Pronoun-Sent', 'Pronoun-Sent-Offset', 'A-Sent', 'A-Sent-Offset', 'B-Sent',
                            'B-Sent-Offset'])


def add_sent_columns(df, text_column, pronoun_offset_column, a_offset_column, b_offset_column):
    global seq_list
    seq_list = list()
    sent_df = df.apply(lambda row: extract_sents(row.loc[text_column], row[pronoun_offset_column], row[a_offset_column],
                                                 row[b_offset_column], row['ID']), axis=1)
    df = df.join(sent_df.set_index('ID'), on='ID')
    return df, seq_list


def sentences_to_sequences(tokenized_, word_index):
    return list(map(
        lambda sent_tokenized: list(map(
            lambda token_: word_index[token_.text] if token_.text in word_index else 0,
            sent_tokenized
        )),
        tokenized_
    ))


def poses_to_sequences(tokenized_, pos_index):
    return list(map(
        lambda sent_tokenized: list(map(
            lambda token_: pos_index[token_.pos] if token_.pos in pos_index else 0,
            sent_tokenized
        )),
        tokenized_
    ))