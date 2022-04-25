

from nltk.tokenize import sent_tokenize, word_tokenize

from math import log

from nltk.stem import PorterStemmer

import nltk

from nltk.corpus import stopwords

from google.colab import drive

import nltk

import os

def raciner(l):
  ps = PorterStemmer()
  a=[]
  for i in l:
    a.append(ps.stem(i))
  return a

def filter(l):
  s=set(stopwords.words("english"))
  a=[]
  for i in l:
    if not i in s:
      a.append(i) 
  return a

def nb_occ(x,l):
  all_words=nltk.FreqDist(l)
  return all_words[x]

def poid(x,l,nb_doc):
  return (1+log(nb_occ(x,l)))*log(nb_doc/contien(l,x))

def doc_retourne(x,l):
  res={}
  for k,v in l.items():
    if nb_occ(x,v)>0:
      res[k]=v
  return res

def poid_doc(x,l):
  doc_poid={}
  for i in l.keys():
    doc_poid[i]=poid(x,l[i],nb_doc)
  return doc_poid

def contien(l,x):
  cont=0
  for i in l.values():
    if x in i:
      cont+=1
  return cont

nltk.download("book")

drive.mount('/content/drive')

l=os.listdir('/content/drive/MyDrive/tp_index')

f_txt=[]
for i in l:
  if i[-3:]=="txt":
    f_txt.append(i)
nb_doc=len(f_txt)

x=input()
x1=x

ps1 = PorterStemmer()
x=ps1.stem(x)

# a tester
txt={}
for i in f_txt:
  a="/content/drive/MyDrive/tp_index/"+i
  f=open(a,'r')
  txt[i]=word_tokenize(f.read())

for i in txt.keys():
  l1=filter(txt[i])
  txt[i]=raciner(txt[i])

txt=doc_retourne(x,txt)

for i in txt.keys():
  print("documen {} nombre d occurance du mot {} avec un poid de {}".format(i,nb_occ(x,txt[i]),poid(x,txt[i],nb_doc)))

#recherche

EXAMPLE_TEXT = "Breast cancer is the most common form of cancer worldwide among women, with a high mortality rate. In fact, in most cases, breast cancer leads to death. Nevertheless, thanks to the early detection , the number of breast cancer related deaths was been reduced in the lastdecade. The best tool to carry out the early breast cancer detection is mammography, where through certain typical signatures like masses and microcalcifications can help in the early diagnosis of this dangerous cancer."
l=word_tokenize(EXAMPLE_TEXT)
l=filter(l)
print(l)
l=raciner(l)
print(l)
print(nb_occ('breast',l))