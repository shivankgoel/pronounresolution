#filter WSJ.pron
import re
import json
#pronoun  = {}

lst = ['her','His', 'his','She', 'she', 'Him','him','He','Her', 'he']

flag  = 0
Antecedent = ""
pronoun = ""
#data statistics
'''
with open("../bbn-pcet/data/BBN-wsj-pronouns/WSJ.pron") as f:
    for content in f:
        content  = content.rstrip()
        content  = content.rstrip(" ")
        content  = re.sub('\s+',' ',content)
        if(content[0:4]=="(WSJ"):
            flag = flag + 1
        elif(conten't[0:11]==" Antecedent"):
            Antecedent = content
        elif(content[0:8]==" Pronoun"):
            pos = content.rfind('>')
            word = content[pos+1:]
            word  = word [1:]
            if word in lst:
                print(word)
                if word in pronoun.keys():
                    pronoun[word] = pronoun[word] + 1
                else:
                    pronoun[word] = 1
        print(pronoun)
'''
data_inputs = []
with open("../bbn-pcet/data/BBN-wsj-pronouns/WSJ.pron") as f:
    for content in f:
        content  = content.rstrip()
        content  = content.rstrip(" ")
        content  = re.sub('\s+',' ',content)
        if(content[0:4]=="(WSJ"):
            flag = flag + 1
        elif(content[0:11]==" Antecedent"):
            Antecedent = content
            Antecedent = Antecedent.split()
            Antecedent = [Antecedent[2],' '.join(Antecedent[4:])]
        elif(content[0:8]==" Pronoun"):
            pos = content.rfind('>')
            word = content[pos+1:]
            word  = word [1:]
            if word in lst:
                pronoun = content
                pronoun = pronoun.split()
                pronoun = [pronoun[2],' '.join(pronoun[4:])]
                temp = []
                temp.append(flag)
                temp.extend(Antecedent)
                temp.extend(pronoun)
                data_inputs.append(temp)

document = {}
temp_lst = []
flag = 0
with open("../bbn-pcet/data/BBN-wsj-pronouns/WSJ.sent") as f:
    for content in f:
        content  = content.rstrip()
        content  = content.rstrip(" ")
        content  = re.sub('\s+',' ',content)
        if(content[0:4]=="(WSJ"):
            flag = flag + 1
            temp_lst = []
        elif(content==")"):
            document[flag] = temp_lst
        else:
            temp_lst.append(content)

data_outputs = []
for i in data_inputs:
    temp = []
    sentence = set()
    pronoun_offset = i[3].split(":")
    A_offset =  i[1].split(":")
    sentence.add(int(pronoun_offset[0][1:]))
    sentence.add(int(A_offset[0][1:]))
    sentence = sorted(sentence)
    text = []
    for j in sentence:
        raw_sentence = document[int(i[0])][j-1]
        raw_sentence = raw_sentence.split(":")
        raw_sentence = ":".join(raw_sentence[1:])
        text.append(raw_sentence[1:])
    text = " ".join(text)
    temp.append(text)
    temp.append(i[4])
    temp.append(pronoun_offset[1].split("-")[0])
    temp.append(i[2])
    temp.append(A_offset[1].split("-")[0])
    data_outputs.append(temp)

with open('WSJ.json', 'w') as outfile:
    json.dump(data_outputs, outfile)

#[1296, 'S66:10-10', 'Smith', 'S68:1-1', 'He']
#Text Pronoun	Pronoun-offset	A	A-offset
