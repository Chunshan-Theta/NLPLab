# encoding:utf-8
#!/usr/bin/python
import  MySQLdb
import logging
logging.basicConfig(level=logging.INFO)
import os
import re




if not os.path.exists("negativeReviews"):
    os.makedirs("negativeReviews")
if not os.path.exists("positiveReviews"):
    os.makedirs("positiveReviews")

def createFile(type,content,title):
    assert type == 1 or type == 2
    if type ==1:
        with open("./negativeReviews/"+str(title)+".txt","w+") as f:
            f.write(content)
    else:
        with open("./positiveReviews/"+str(title)+".txt","w+") as f:
            f.write(content)

db  =  MySQLdb.connect (
     host = "140.115.126.20" ,     #主機名
     user = "theta" ,          #用戶名
     passwd = "theta" ,   #密碼
     db = "sentiment" )         #數據庫名稱

#查詢前，必須先獲取游標
cur  =  db.cursor()
cur.execute('SET NAMES UTF8')
#執行的都是原生SQL語句
cur.execute( "SELECT `action_id`,`content`,`sayType` FROM `action_list` WHERE `sayType`!= 0" )

for  row  in  cur.fetchall():
    logging.debug(row[1])
    content = row[1]
    content = re.sub("[A-Za-z0-9]+", "", content) # remove English and number
    content = re.sub("[/.~?=,]+", "", content) # remove Any character symbols
    logging.info(content)
    if len(content)>50:
        createFile(2,content,row[0])
    else:
        createFile(1,content,row[0])


db.close()
