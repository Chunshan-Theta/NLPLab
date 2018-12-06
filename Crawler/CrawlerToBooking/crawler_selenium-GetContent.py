#coding:utf-8
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import datetime
import os
from os import listdir
from os.path import isfile, join
from io import open

ChromeDriveDir ="/usr/lib/chromium-browser/chromedriver"
driver = webdriver.Chrome(ChromeDriveDir)

if not os.path.exists("negativeReviews"):
    os.makedirs("negativeReviews")
if not os.path.exists("positiveReviews"):
    os.makedirs("positiveReviews")




def record(text,name,folder):
    f = open("./"+folder+"/"+name+".txt","w+")
    f.write(text)
    f.write(unicode('\n'))
    f.close()
    print(name)

def logged(text):
    print("logged",text)
    print("-"*20)
    f2 = open("./log/url","a+")
    f2.write(text)
    f2.write(unicode('\n'))
    f2.close()


f = open("./log/url","r")
urlLogSet=[]
for line in f.readlines():

    urlLogSet.append(line)
f.close()

urlSet=[]
urlFiles = ['url/' + f for f in listdir('url/') if isfile(join('url/', f))]
for uf in urlFiles:
    with open(uf, "r", encoding='utf-8') as f:
        for line in f.readlines():
            urlSet.append(line)

for rootUrlIdx in range(len(urlSet)):


    rootUrl = urlSet[rootUrlIdx]
    print(rootUrl)
    if rootUrl in urlLogSet:
        print("in urlLogSet")
        continue
    else:
        driver.get("https://www.booking.com/reviews/tw/hotel/"+rootUrl)
        title = driver.find_element_by_class_name("standalone_header_hotel_link").text
        count_p=0
        count_n=0
        logged(rootUrl)

        while 1:
            driver.implicitly_wait(10)
            review_neg_divs = driver.find_elements_by_class_name("review_neg")
            review_pos_divs = driver.find_elements_by_class_name("review_pos")

            for i in review_pos_divs:
                count_p+=1
                print(count_p)
                record(i.text,title+str(count_p),"positiveReviews")

            for i in review_neg_divs:
                count_n+=1
                print(count_n)
                record(i.text,title+str(count_n),"negativeReviews")


            try:
                PageNextLink = driver.find_element_by_id("review_next_page_link").get_attribute('href')
                #print(PageNextLink)
                driver.get(str(PageNextLink))
            except Exception as e:
                print(e)
                break


#driver.close()
