import csv
import gender_guesser.detector as gender

d = gender.Detector()

lst = ['her','Her','She','she']
num = 0
data = []
with open('gap-test.tsv') as tsvfile:
    reader = csv.DictReader(tsvfile,dialect='excel-tab')
    for row in reader:
        pron = row['Pronoun']
        if pron in lst:
            pron = "female"
        else:
            pron = "male"
        A = row['A']
        acoref = row['A-coref']
        B = row['B']
        bcoref = row['B-coref']

        if acoref == "TRUE":
            typea = pron
            typeb = d.get_gender(B.split()[0])
        else:
            typeb = pron
            typea = d.get_gender(A.split()[0])
        data.append([typea,typeb])

with open('test_gender.tsv', mode='w') as csv_file:
    fieldnames = ['a_gender','b_gender']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames,dialect='excel-tab')
    writer.writeheader()
    for record in data:
        writer.writerow({'a_gender':record[0],'b_gender': record[1]})
