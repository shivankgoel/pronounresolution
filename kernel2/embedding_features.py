from dependencies import *


def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
    else:
        return node.orth_


def bs(list_, target_):
    lo, hi = 0, len(list_) - 1

    while lo < hi:
        mid = lo + int((hi - lo) / 2)

        if target_ < list_[mid]:
            hi = mid
        elif target_ > list_[mid]:
            lo = mid + 1
        else:
            return mid + 1
    return lo


def _get_preceding_words(tokens, offset, k):
    start = offset - k

    precedings = [None] * max(0, 0 - start)
    start = max(0, start)
    precedings += tokens[start: offset]

    return precedings


def _get_following_words(tokens, offset, k):
    end = offset + k

    followings = [None] * max(0, end - len(tokens))
    end = min(len(tokens), end)
    followings += tokens[offset: end]

    return followings


def embedding_features(text, char_offset):
    doc = nlp(text)

    # char offset to token offset
    lens = [token.idx for token in doc]
    mention_offset = bs(lens, char_offset) - 1
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

    # dependency parent
    head = mention.head

    # last word and first word
    first_word, last_word = sent[0], sent[-2]

    assert mention_offset >= 0

    # two preceding words and two following words
    tokens = list(doc)
    precedings2 = _get_preceding_words(tokens, mention_offset, 2)
    followings2 = _get_following_words(tokens, mention_offset, 2)

    # five preceding words and five following words
    precedings5 = _get_preceding_words(tokens, mention_offset, 5)
    followings5 = _get_following_words(tokens, mention_offset, 5)

    # sentence words
    sent_tokens = [token for token in sent]

    return mention, head, first_word, last_word, precedings2, followings2, precedings5, followings5, sent_tokens
