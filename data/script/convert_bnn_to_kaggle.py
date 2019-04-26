import nltk
import csv
from random import sample
import kaggle_format as kg

def test(tmp):
    if(tmp.text[tmp.pronoun_offset:tmp.pronoun_offset+len(tmp.pronoun)] != tmp.pronoun):
        print(tmp.text[tmp.pronoun_offset:tmp.pronoun_offset+len(tmp.pronoun)],tmp.pronoun)
        print("error1")
    if(tmp.text[tmp.a_offset:tmp.a_offset+len(tmp.a)] != tmp.a):
        print(tmp.text[tmp.a_offset:tmp.a_offset+len(tmp.a)],tmp.a)
        print("error2")
    if(tmp.text[tmp.b_offset:tmp.b_offset+len(tmp.b)] != tmp.b):
        print(tmp.text[tmp.b_offset:tmp.b_offset+len(tmp.b)],tmp.b)
        print("error3")

#TODO:find nnp in the sentence
def getnnp(s):
    s = nltk.word_tokenize(s)
    s = nltk.pos_tag(s)
    grammar = "person: {<NNP>+}"
    chunker = nltk.chunk.RegexpParser(grammar)
    t = chunker.parse(s)
    lst = []
    for i in t.subtrees(filter=lambda x: x.label() == 'person'):
        str = []
        for word in i:
            str.append(word[0])
        str = " ".join(str)
        lst.append(str)
    return lst

data = []
id = 0
with open('bnn_original.csv') as csvfile:
    reader = csv.DictReader(csvfile,dialect='excel-tab')
    for row in reader:
        text = row["Text"]
        text_lst = nltk.sent_tokenize(text)
        pronoun = row["Pronoun"]
        a = row["A"]
        length = 0
        a_offset = None
        pronoun_offset = None
        for i in text_lst:
            sent = i.split(" ")
            if int(row["Pronoun-offset"]) > len(sent):
                length = length + len(i) + 1
                continue
            if sent[int(row["Pronoun-offset"])-1] == pronoun:
                pronoun_offset = length + sum([len(sent[x])+1 for x in range(int(row["Pronoun-offset"])-1)])
                break;
            else:
                length = length + len(i) + 1
        if pronoun_offset == None:
            continue
        length = 0
        for i in text_lst:
            sent = i.split(" ")
            if int(row["A-offset"]) > len(sent):
                length = length + len(i) + 1
                continue
            if sent[int(row["A-offset"])-1] == a:
                a_offset = length + sum([len(sent[x])+1 for x in range(int(row["A-offset"])-1)])
                break;
            else:
                length = length + len(i) + 1
        if a_offset == None:
            continue
        b = ""
        b_candidate = getnnp(text)
        #very importent step: shuffle here
        b_candidate = sample(b_candidate, len(b_candidate))
        for i in b_candidate:
            if(i!=pronoun and i!=a):
                b = i
                break
        b_offset = text.index(b)
        if a_offset < b_offset:
            tmp = kg.Kaggledata("bnn"+str(id),text,pronoun,pronoun_offset,a,a_offset,"TRUE",
            b,b_offset,"FALSE",'unk')
            test(tmp)
            data.append(tmp)
        else:
            tmp = kg.Kaggledata("bnn"+str(id),text,pronoun,pronoun_offset,b,b_offset,"FALSE",
            a,a_offset,"TRUE",'unk')
            test(tmp)
            data.append(tmp)
        id = id + 1

with open('bnn_kaggleformat.csv', mode='w') as csv_file:
    fieldnames = ['ID','Text','Pronoun','Pronoun-offset','A','A-offset','A-coref','B','B-offset','B-coref','URL']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames,dialect='excel-tab')
    writer.writeheader()
    for record in data:
        writer.writerow({'ID':record.id,'Text': record.text, 'Pronoun': record.pronoun, 'Pronoun-offset': str(record.pronoun_offset),
        'A':record.a, 'A-offset':str(record.a_offset),'A-coref':record.a_coref,
        'B':record.b, 'B-offset':str(record.b_offset),'B-coref':record.b_coref,'URL':record.url})
