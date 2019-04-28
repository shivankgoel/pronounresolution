from random import shuffle
import kaggle_format as kg
import csv

data = []
female = ['her','She', 'she', 'Her']
male = ['His', 'his', 'Him','him','He', 'he']
with open('bnn_kaggleformat.csv') as tsvfile:
    reader = csv.DictReader(tsvfile,dialect='excel-tab')
    for row in reader:
        tmp = kg.Kaggledata(row['ID'],row['Text'],row['Pronoun'],int(row['Pronoun-offset']),row['A'],int(row['A-offset']),row['A-coref'],
        row['B'],int(row['B-offset']),row['B-coref'],row['URL'])
        data.append(tmp)

shuffle(data)
validation = []
num_male_val = 0
num_female_val = 0
test = []
num_male_test = 0
num_female_test = 0
train = []

for i in data:
    if i.pronoun in female:
        if(num_female_test < 120):
            test.append(i)
            num_female_test = num_female_test + 1
        elif(num_female_val < 62):
            validation.append(i)
            num_female_val = num_female_val + 1
        else:
            train.append(i)
    else:
        if(num_male_test < 120):
            test.append(i)
            num_male_test = num_male_test + 1
        elif(num_male_val < 62):
            validation.append(i)
            num_male_val = num_male_val + 1
        else:
            train.append(i)

with open('bnn_train.csv', mode='w') as csv_file:
    fieldnames = ['ID','Text','Pronoun','Pronoun-offset','A','A-offset','A-coref','B','B-offset','B-coref','URL']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames,dialect='excel-tab')
    writer.writeheader()
    for record in train:
        writer.writerow({'ID':record.id,'Text': record.text, 'Pronoun': record.pronoun, 'Pronoun-offset': str(record.pronoun_offset),
        'A':record.a, 'A-offset':str(record.a_offset),'A-coref':record.a_coref,
        'B':record.b, 'B-offset':str(record.b_offset),'B-coref':record.b_coref,'URL':record.url})

with open('bnn_test.csv', mode='w') as csv_file:
    fieldnames = ['ID','Text','Pronoun','Pronoun-offset','A','A-offset','A-coref','B','B-offset','B-coref','URL']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames,dialect='excel-tab')
    writer.writeheader()
    for record in test:
        writer.writerow({'ID':record.id,'Text': record.text, 'Pronoun': record.pronoun, 'Pronoun-offset': str(record.pronoun_offset),
        'A':record.a, 'A-offset':str(record.a_offset),'A-coref':record.a_coref,
        'B':record.b, 'B-offset':str(record.b_offset),'B-coref':record.b_coref,'URL':record.url})

with open('bnn_validation.csv', mode='w') as csv_file:
    fieldnames = ['ID','Text','Pronoun','Pronoun-offset','A','A-offset','A-coref','B','B-offset','B-coref','URL']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames,dialect='excel-tab')
    writer.writeheader()
    for record in validation:
        writer.writerow({'ID':record.id,'Text': record.text, 'Pronoun': record.pronoun, 'Pronoun-offset': str(record.pronoun_offset),
        'A':record.a, 'A-offset':str(record.a_offset),'A-coref':record.a_coref,
        'B':record.b, 'B-offset':str(record.b_offset),'B-coref':record.b_coref,'URL':record.url})
