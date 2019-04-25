import csv


class Kaggledata:
  def __init__(self, id,text,pronoun,pronoun_offset,a,a_offset,a_coref,b,b_offset,b_coref,url):
    self.id = id
    self.text = text
    self.pronoun = pronoun
    self.pronoun_offset = pronoun_offset
    self.a = a
    self.a_offset = a_offset
    self.a_coref = a_coref
    self.b = b
    self.b_offset = b_offset
    self.b_coref = b_coref
    self.url = url

lst = ['her','His', 'his','She', 'she', 'Him','him','He','Her', 'he']
oppose_lst = ['his/him','Her', 'her','He', 'he', 'Her','her','She','His/Him', 'she']
data = []

def replacesmallest(a,b,c,tmp):
    if a<b and a<c:
        len_pronoun = len(tmp.pronoun)
        prev = tmp.text[0:tmp.pronoun_offset]
        after = tmp.text[tmp.pronoun_offset+len_pronoun:]
        reverse_pronoun = oppose_lst[lst.index(tmp.pronoun)]
        diff = len(tmp.pronoun) - len(reverse_pronoun)
        tmp.text = prev + reverse_pronoun + after
        tmp.pronoun = reverse_pronoun
        return diff
    elif b<a and b<c:
        len_word = len(tmp.a)
        prev = tmp.text[0:tmp.a_offset]
        after = tmp.text[tmp.a_offset+len_word:]
        reverse = "E1"
        diff = len(tmp.a) - 2
        tmp.text = prev + reverse + after
        tmp.a = "E1"
        return diff
    else:
        len_word = len(tmp.b)
        prev = tmp.text[0:tmp.b_offset]
        after = tmp.text[tmp.b_offset+len_word:]
        reverse = "E2"
        diff = len(tmp.b) - 2
        tmp.text = prev + reverse + after
        tmp.b = "E2"
        return diff

def replacesecond(a,b,c,tmp,changenum):
    if (a<b and a>c) or (a<c and a>b):
        tmp.pronoun_offset = tmp.pronoun_offset - changenum
        len_pronoun = len(tmp.pronoun)
        prev = tmp.text[0:tmp.pronoun_offset]
        after = tmp.text[tmp.pronoun_offset+len_pronoun:]
        reverse_pronoun = oppose_lst[lst.index(tmp.pronoun)]
        diff = len(tmp.pronoun) - len(reverse_pronoun)
        tmp.text = prev + reverse_pronoun + after
        tmp.pronoun = reverse_pronoun
        return diff
    elif (b<a and b>c) or (b<c and b>a):
        tmp.a_offset = tmp.a_offset - changenum
        len_word = len(tmp.a)
        prev = tmp.text[0:tmp.a_offset]
        after = tmp.text[tmp.a_offset+len_word:]
        reverse = "E1"
        diff = len(tmp.a) - 2
        tmp.text = prev + reverse + after
        tmp.a = "E1"
        return diff
    else:
        tmp.b_offset = tmp.b_offset - changenum
        len_word = len(tmp.b)
        prev = tmp.text[0:tmp.b_offset]
        after = tmp.text[tmp.b_offset+len_word:]
        reverse = "E2"
        diff = len(tmp.b) - 2
        tmp.text = prev + reverse + after
        tmp.b = "E2"
        return diff

def replacelast(a,b,c,tmp,changenum):
    if a>b and a>c:
        tmp.pronoun_offset = tmp.pronoun_offset - changenum
        len_pronoun = len(tmp.pronoun)
        prev = tmp.text[0:tmp.pronoun_offset]
        after = tmp.text[tmp.pronoun_offset+len_pronoun:]
        reverse_pronoun = oppose_lst[lst.index(tmp.pronoun)]
        tmp.text = prev + reverse_pronoun + after
        tmp.pronoun = reverse_pronoun
    elif b>a and b>c:
        tmp.a_offset = tmp.a_offset - changenum
        len_word = len(tmp.a)
        prev = tmp.text[0:tmp.a_offset]
        after = tmp.text[tmp.a_offset+len_word:]
        reverse = "E1"
        tmp.text = prev + reverse + after
        tmp.a = "E1"
    else:
        tmp.b_offset = tmp.b_offset - changenum
        len_word = len(tmp.b)
        prev = tmp.text[0:tmp.b_offset]
        after = tmp.text[tmp.b_offset+len_word:]
        reverse = "E2"
        tmp.text = prev + reverse + after
        tmp.b = "E2"
def test(tmp):
    if(tmp.text[tmp.pronoun_offset:tmp.pronoun_offset+len(tmp.pronoun)] != tmp.pronoun):
        print("error1")
    if(tmp.text[tmp.a_offset:tmp.a_offset+len(tmp.a)] != tmp.a):
        print("error2")
    if(tmp.text[tmp.b_offset:tmp.b_offset+len(tmp.b)] != tmp.b):
        print("error3")

with open('../kaggle/gap-development.tsv') as tsvfile:
    reader = csv.DictReader(tsvfile,dialect='excel-tab')
    for row in reader:
        tmp = Kaggledata(row['ID'],row['Text'],row['Pronoun'],int(row['Pronoun-offset']),row['A'],int(row['A-offset']),row['A-coref'],
        row['B'],int(row['B-offset']),row['B-coref'],row['URL'])
        #TODO:gender swapping
        change0 = replacesmallest(tmp.pronoun_offset,tmp.a_offset,tmp.b_offset,tmp)
        change1 = replacesecond(tmp.pronoun_offset,tmp.a_offset,tmp.b_offset,tmp,change0)
        replacelast(tmp.pronoun_offset,tmp.a_offset,tmp.b_offset,tmp,change0+change1)
        test(tmp)
        data.append(tmp)

with open("anonymous_kaggle.tsv", "w") as record_file:
    record_file.write("ID	Text	Pronoun	Pronoun-offset	A	A-offset	A-coref	B	B-offset	B-coref	URL\n")
    for record in data:
        record_file.write(record.id+"  "+record.text+"  "+ record.pronoun+
        " "+str(record.pronoun_offset)+" "+record.a+" "+str(record.a_offset)+" "+record.a_coref+" "+
        record.b+" "+str(record.b_offset)+" "+record.b_coref+" "+record.url + "\n")
