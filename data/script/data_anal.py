import csv
pronoun  = {}


lst = [['her','Her'], ['She','she'] , ['His', 'his','him'] , ['he','He']]

with open('bnn_train.csv') as tsvfile:
    reader = csv.DictReader(tsvfile,dialect='excel-tab')
    for row in reader:
        temp_pron = row['Pronoun']
        if temp_pron in pronoun.keys():
            pronoun[temp_pron] = pronoun[temp_pron] + 1
        else:
            pronoun[temp_pron] = 1

print(pronoun)

for i in lst:
    print(sum(pronoun[x] for x in i))
