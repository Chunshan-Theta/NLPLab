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
    print(w1+','+w2)
    print(Distance(location(w1),location(w2)))


print('loading json Data .......')
Json_file = open("./output/outputWord2Vec(v300)(50000).txt","r")
Json_file = json.load(Json_file)
'''
print(d)
print(d[u'\u5c07\u4e4b\u652c'])
print(u'\u5c07\u4e4b\u652c')
'''

print('calculate distance .......')

Distance_word("害怕","恐懼")
Distance_word("害怕","畏懼")
Distance_word("害怕","喜歡")
Distance_word("害怕","安全")
