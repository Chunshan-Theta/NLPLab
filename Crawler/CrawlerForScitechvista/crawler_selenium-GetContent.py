#coding:utf-8
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import datetime

ChromeDriveDir ="/usr/lib/chromium-browser/chromedriver"
driver = webdriver.Chrome(ChromeDriveDir)





def record(text,name):

    f = open("./DownloadText/"+name+".txt","aw+")
    f.write(text.encode('utf-8'))
    f.write('\n')
    f.close()
    print(name)

def logged(text,name):
    print("logged",text)
    print("-"*20)
    f2 = open("./log/url","aw+")
    f2.write(text.encode('utf-8'))
    f2.write(name.encode('utf-8')+'\n')
    f2.close()


f = open("./log/url","r")
urlLogSet=[]
for line in f.readlines():

    urlLogSet.append(line)
f.close()

f = open("url","r")
urlSet=[]
for line in f.readlines():
    #print(line)
    urlSet.append(line)
f.close()
for rootUrlIdx in range(len(urlSet)):


    rootUrl = urlSet[rootUrlIdx]
    print(rootUrl)
    if rootUrl in urlLogSet:
        print("in urlLogSet")
        continue
    try:
        driver.get(rootUrl)

        driver.implicitly_wait(10)
        title = driver.find_element_by_tag_name("h1").text
        maindive = driver.find_element_by_class_name("reset_table")
        fillterText = ""
        for subtext in maindive.text.split(" "):
            if len(subtext)>20 and not subtext.encode('utf-8').find("圖片來源：")>-1:
                fillterText+=" "+subtext

        record(fillterText,title)
        logged(rootUrl,title)
    except(Exception):
        pass

#driver.close()
