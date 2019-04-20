#filter WSJ.pron
import re
import json
#pronoun  = {}

lst = ['her','His', 'his','She', 'she', 'Him','him','He','Her', 'he']

flag  = 0
Antecedent = ""
pronoun = {}

#data statistics
with open("../bbn-pcet/data/BBN-wsj-pronouns/WSJ.pron") as f:
    for content in f:
        content  = content.rstrip()
        content  = content.rstrip(" ")
        content  = re.sub('\s+',' ',content)
        if(content[0:4]=="(WSJ"):
            flag = flag + 1
        elif(content[0:11]==" Antecedent"):
            Antecedent = content
        elif(content[0:8]==" Pronoun"):
            pos = content.rfind('>')
            word = content[pos+1:]
            word  = word [1:]
            if word in lst:
                if word in pronoun.keys():
                    pronoun[word] = pronoun[word] + 1
                else:
                    pronoun[word] = 1
for i in lst:
    if i not in pronoun.keys():
        pronoun[i] = 0
print(pronoun)
