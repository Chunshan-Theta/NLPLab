# -*- coding:utf-8 -*-
#!usr/bin/env python
import json

Json_file = open("./output(10day).txt","r")
Json_file = json.load(Json_file)
'''
print(d)
print(d[u'\u5c07\u4e4b\u652c'])
print(u'\u5c07\u4e4b\u652c')
'''
def location(text):
    uni_text=unicode(text,"utf-8")
    try:
        #print(Json_file[uni_text])
        return Json_file[uni_text]
    except KeyError :
        #print(uni_text)
        return 
def diff(t1,t2):
    x = t1[0]-t2[0]
    y = t1[1]-t2[1]
    print(x,y,((x**2+y**2)**0.5))

def diff_word(w1,w2):
    print(w1+','+w2)
    print(diff(location(w1),location(w2)))





diff_word("然後","即使")
diff_word("台灣","英國")
diff_word("法國","英國")

