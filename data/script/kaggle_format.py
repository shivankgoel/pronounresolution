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
    
