#filter WSJ.pron
import re
pronoun  = {}

lst = ['her','His', 'his','She', 'she', 'Him','him','He','Her', 'he']

flag  = 0
Antecedent = ""
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
                print(word)
                if word in pronoun.keys():
                    pronoun[word] = pronoun[word] + 1
                else:
                    pronoun[word] = 1
print(pronoun)
