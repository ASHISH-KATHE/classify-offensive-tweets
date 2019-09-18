# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 20:18:48 2019

@author: Ashish
"""
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
ww=[]
wwi=[]
def check(wrd):
    for word in ww:
        if word == wrd:
            return 1
    return 0

import spacy
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
p_stem=SnowballStemmer(language='english')

df=pd.read_csv('new_data.csv')

print('spaCy Version: %s' % (spacy.__version__))
spacy_nlp = spacy.load('en_core_web_sm')
y=[]
y=df['label']
df_fin=[]
df_fin.append([])
s_stop= spacy.lang.en.stop_words.STOP_WORDS
s_stop.add('')
print('Number of stop words: %d' % len(s_stop))
#string='This story is about surfing Catching waves is fun Surfing is a popular water sport.'
#print(string)
str2=[]
k=0
for str1 in df['tweet']:
    str1=str1.lower()
    str2.clear()
    str1=str1.split()
    i=len(str1)
    j=0
    while j<i:
        if str1[j] in s_stop:
            str1.remove(str1[j])
            i=i-1
            continue
        j=j+1
    
    for word in str1:
        str2.append(p_stem.stem(word))
    length=len(str2)
    temp=0
    while temp<length:
        str2[temp] = "".join(c for c in str2[temp] if c not in ('!','.',':',',','?','@'))
        if ps.stem(str2[temp]) in badword_txt and len(ps.stem(str2[temp]))>2:
            df_fin[k].append(ps.stem(str2[temp]))
        temp=temp+1
    df_fin.append([])
    k=k+1
    str1.clear()
df_fin.pop()
from sklearn.model_selection import train_test_split

#x_train,x_test,y_train,y_test=train_test_split(df_fin,df['label'],test_size=0.33)

badword=open('badwords.txt')
badword_txt=badword.read().split('\n')
temp=0
for word in df_fin:
    for let in word:
        if check(let)==1:
            if y[temp]==0:
                if let in badword_txt:
                    wwi[ww.index(let)]+=0.08
            elif y[temp]==1:
                if let in badword_txt:
                    wwi[ww.index(let)]-=0.18
                    
        elif let in badword_txt:
            ww.append(let)
            wwi.append(0)
    temp+=1



sw=[]
temp=0
for word in df_fin:
    sw.append(0)
    for let in word:
        sw[temp]+=wwi[ww.index(let)]
    temp+=1
y_pred=[]
'''summ=0
summ1=0
temp=0
max1=0
min1=0
temp1=0
temp2=0
for x in sw:
    if y[temp]==0:
        summ+=x
        if x<min1:
            min1=x
        temp1+=1
    elif y[temp]==1:
        summ1+=x
        if x>max1:
            max1=x
        temp2+=1
    temp+=1'''
cd=[]
for x in sw:
    if x>10:
        y_pred.append(0)
    elif x<0:
        y_pred.append(1)
    else :
        y_pred.append(0)


temp=0
temp1=0
while temp<len(y):
    if y[temp]==1 and y_pred[temp]==0:
        cd.append(sw[temp])
    temp+=1

'''print(summ/temp1)
print(summ1/temp2)
print(max1)
print(min1)'''
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, y_pred)

print(cm)
'''for str1 in df['tweet']:
    str1=str1.lower()
    str2.clear()
    str1=str1.split()
    i=len(str1)
    j=0
    while j<i:
        if str1[j] in s_stop:
            str1.remove(str1[j])
            i=i-1
            continue
        j=j+1
    length=len(str2)
    temp=0
    while temp<length:
        df_fin[k].append(str2[temp])
        temp=temp+1
    df_fin.append([])
    k=k+1
    str1.clear()
df_fin.pop()'''


