# -*- coding:utf-8 -*-
#!usr/bin/env python
import json



def location(text):
    #uni_text=unicode(text,"utf-8")
    uni_text=text
    try:
        #print(Json_file[uni_text])
        return Json_file[uni_text]
    except KeyError :
        #print(uni_text)
        return
def Distance(t1,t2):

    assert len(t1) == len(t2)
    sum=0
    for i in range(len(t1)):
        sum+=((t1[i]-t2[i])**2)


    #x = t1[0]-t2[0]
    #y = t1[1]-t2[1]
    #return ((x**2+y**2)**0.5)*10000000000
    return (sum**0.5)*100

def Distance_word(w1,w2):
    #print(w1+','+w2)
    #print(Distance(location(w1),location(w2)))
    return Distance(location(w1),location(w2))


print('loading json Data .......')
Json_file = open("./outputWord2Vec(v300)(0).txt","r")
Json_file = json.load(Json_file)
print('done')
'''
print(d)
print(d[u'\u5c07\u4e4b\u652c'])
print(u'\u5c07\u4e4b\u652c')
'''

'''
Distance_word("害怕","恐懼")
Distance_word("害怕","畏懼")
Distance_word("害怕","喜歡")
Distance_word("害怕","娛樂")
'''

relation1=[
  ["核能","核子"],
  ["核電廠","核子"],
  ["電力","核電廠"],
  ["核電廠","核能"],
  ["核能","核子"],
  ["核能發電","火力發電"],
  ["基改","基因"],
  ["基改","基因改良"],
  ["基改食品","基改"],
  ["基改作物","基改食品"],
  ["基因工程","基因改造"],
  ["基改","基因工程"]
]
relation0=[
  ["核能","基因"],
  ["核電廠","基因改良"],
  ["電力","基改"],
  ["核電廠","基改食品"],
  ["核能","基因改造"],
  ["核能發電","基因工程"],
  ["核能發電","基因"],
  ["核能","基因改良"],
  ["核電廠","基改"],
  ["電力","基改食品"],
  ["核電廠","基因改造"],
  ["核能","基因工程"]
]

sum = 0
for i in relation0:
    unit = Distance_word(i[0],i[1])
    sum+=unit
print(sum/len(relation0))
print('-'*20)
sum = 0
for i in relation1:
    unit = Distance_word(i[0],i[1])
    sum+=unit
print(sum/len(relation1))
