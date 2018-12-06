#coding:utf-8
import requests as rq
import datetime
import time
import random





def catch_row_from_list(News_link_list):
    try:
        f_start = News_link_list.index('<h2>')
        f_End = News_link_list.index('</h2>')+5

        row = News_link_list[f_start:f_End]
        #print(row)
        row_start = row.index('<a href="')+9
        row_End = row.index('" class')
        link = row[row_start:row_End]
        news_link_daily.append(link)
        #print(news_link_daily)
        catch_row_from_list(News_link_list[f_End:])
    except ValueError as e:

        print('page end',random.randint(0,9))
    except Exception as e:
        print "Unknow error:"+str(e) 




def changePage(root_class,numpage=1):
    Page = rq.get('http://www.chinatimes.com/history-by-date/'+root_class+'?page='+str(numpage))#目標網佔
    content = Page.content
    content_start = Page.content.index('<div class="listRight">')
    content_End = Page.content.index('class="pagination clear-fix"')
    News_link_list = content[content_start:content_End]
    if(len(News_link_list)>130):
        catch_row_from_list(News_link_list)
        changePage(root_class,int(numpage)+1)
    else:
        print('daily class end')





def catch_text_from_page(Text_area):

    try:
        f_Text_area_start = Text_area.index('<p>')+3
        f_Text_area_End =Text_area.index('</p>')
        f_Text_area = Text_area[f_Text_area_start:f_Text_area_End]
        f.write(f_Text_area)
        catch_text_from_page(Text_area[f_Text_area_End+3:])
    except ValueError as e:
        pass
        #print('Text end',random.randint(0,9))
    except Exception as e:
        print "Unknow error:"+str(e) 


def dailynews_d(Date):
    global news_link_daily
    news_link_daily=[]
    #Download link
    for num_c in range(4):
        changePage(Date+'-260'+str(num_c+1))


    #Download text
    news_link_daily_len = len(news_link_daily)
    for i in range(news_link_daily_len):
        try:
            print(str(i)+"/"+str(news_link_daily_len)+"pages")
            SiteRoot='http://www.chinatimes.com'
            link = news_link_daily[i]
            Page = rq.get(SiteRoot+link)#目標網佔
            content = Page.content
            content_start = content.index('<article class="arttext marbotm')
            content_End = content.index('<div class="nav-below2017 marbotm">')
            Text_area = content[content_start:content_End]
            catch_text_from_page(Text_area)
        except Exception as e:
            print("Unknow error:"+str(e))
            print("error link:",news_link_daily[i]) 


    

## global var defined
f= open('result.txt','w')
news_link_daily=[]


# //global
# 
# 
def change_date(needNumDay,starttime=datetime.datetime.today(),NumDay=0):
    if NumDay>needNumDay:
        print('end')
    else:
        T_day = starttime-datetime.timedelta(days=1)
        T_day_str = T_day.strftime('%Y-%m-%d')
        print(T_day_str)
        dailynews_d(T_day_str)
        change_date(needNumDay,T_day,NumDay+1)


p_start = str(datetime.datetime.now())
change_date(360)
p_end = str(datetime.datetime.now())
print(p_start,p_end)
f.close()