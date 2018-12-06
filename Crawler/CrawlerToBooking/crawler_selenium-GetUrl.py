#coding:utf-8
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import time
import datetime

ChromeDriveDir ="/usr/lib/chromium-browser/chromedriver"
driver = webdriver.Chrome(ChromeDriveDir)
#https://www.booking.com/searchresults.zh-tw.html?&ss=臺灣


'''
def workday():
    driver = webdriver.Chrome(ChromeDriveDir)
    #    go to page
    driver.get("https://cis.ncu.edu.tw/HumanSys/login")
    time.sleep(WaitLoadTime)
    elem = driver.find_element_by_id("userid_input")
    elem.send_keys('106524018')
    elem = driver.find_element_by_name("j_password")
    elem.send_keys('gavin840511')

    elem.send_keys(Keys.RETURN)
    time.sleep(WaitLoadTime)
    driver.get("https://cis.ncu.edu.tw/HumanSys/student/stdSignIn")
    time.sleep(WaitLoadTime)

    elem = driver.find_element_by_id("table1")


    link_list = []
    for row in elem.find_elements_by_tag_name("tr"):
        col_list = row.find_elements_by_tag_name("td")
        if(len(col_list)):
            for i in col_list:
                for i2 in i.find_elements_by_tag_name("a"):
                    print(i2.text)
                    if(i2.text.encode('utf-8')=="新增簽到"):
                        link = i2.get_attribute("href")
                        print(link)
                        link_list.append(link)

    #nextturnwaittime = workhoursdaliy*60*60
    for i in link_list:

        driver.get(i)

        try: #sign in
            elem = driver.find_element_by_id("signin")
            elem.click()
        except Exception as e:
            #print(e)
            try: # sign out
                elem = driver.find_element_by_id("AttendWork")
                elem.send_keys('coding works')
                elem = driver.find_element_by_id("signout")
                elem.click()
                #nextturnwaittime = 0#16*60*60
            except Exception as e2:
                print(e)
    driver.close()
'''
'''
	if not nextturnwaittime == 0 :
		print("Next Turn:",nextturnwaittime/3600,"hrs")
		driver.close()
		time.sleep(nextturnwaittime)
	else:
		pass
'''

#id="signin"
#id="signout"
#id="AttendWork"
'''
#    enter email & password
elem = driver.find_element_by_name("email")
elem.clear()
elem.send_keys(FaceBookID)
elem = driver.find_element_by_name("pass")
elem.clear()
elem.send_keys(FaceBookPass)
elem.send_keys(Keys.RETURN)
time.sleep(WaitLoadTime)
'''

def record(text,title=""):
    assert isinstance(title, str)
    assert isinstance(text, str)

    f = open("./url/"+title,"aw+")
    f.write(text)
    f.write('\n')
    f.close()

rootUrlSet = ["https://www.booking.com/searchresults.zh-tw.html?&ss=臺灣"
              ]

for rootUrl in rootUrlSet:
    print(rootUrl)
    tag = rootUrl[rootUrl.find("=")+1:]
    driver.get(rootUrl)
    count=0
    while 1:
        count+=1
        driver.implicitly_wait(10)
        fieldsetBoxlist = driver.find_elements_by_class_name("hotel_name_link")
        for fieldsetBox in fieldsetBoxlist:
            a = fieldsetBox
            link = str(a.get_attribute('href'))
            substring_start = link.find("/tw/")+4
            substring_end = link.find("?label",substring_start)
            link = link[substring_start:substring_end]
            print(link)
            record(link,tag+str(count))
        #rdpActionButton rdpPageNext
        try:
            PageNextLink = driver.find_element_by_class_name("paging-next").get_attribute('href')
            driver.get(PageNextLink)
        except:
            break


driver.close()
