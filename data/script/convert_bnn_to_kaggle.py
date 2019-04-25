import anony as kg
import nltk
import csv


with open('bnn_original.tsv') as tsvfile:
    reader = csv.DictReader(tsvfile,dialect='excel-tab')
    for row in reader:
        text = row["Text"]
        text_lst = nltk.sent_tokenize(text)
        pronoun = row["Pronoun"]
        length = 0
        for i in text_lst:
            sent = i.split(" ")
            print(row["Pronoun"])
            print(row["Pronoun-offset"])
            if sent[int(row["Pronoun-offset"])] == pronoun:
                pronoun_offset = length + sum([len(sent[x])+1 for x in range(int(row["Pronoun-offset"]))])
                break;
            else:
                length = length + len(i) + 1
        print(pronoun)
        print(text[pronoun_offset:])
        break;
