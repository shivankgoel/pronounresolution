import csv
pronoun  = {}
with open('../bbn-pcet/bnn_kaggleformat.csv') as tsvfile:
    reader = csv.DictReader(tsvfile,dialect='excel-tab')
    for row in reader:
        temp_pron = row['Pronoun']
        if temp_pron in pronoun.keys():
            pronoun[temp_pron] = pronoun[temp_pron] + 1
        else:
            pronoun[temp_pron] = 1

print(pronoun)
