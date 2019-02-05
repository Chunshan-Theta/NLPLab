import os
import sys
sys.path.append('../../../jieba_zn/')
import jieba
import collections
import logging
logging.basicConfig(level=logging.INFO)
import re





#读取停用词
stop_words = []
with open('stop_words.txt',"r") as f:
    line = f.readline()
    while line:
        stop_words.append(line[:-1])
        line = f.readline()
stop_words = list(stop_words)
print('停用詞讀取完畢，共{n}個詞'.format(n=len(stop_words)))


# 读取文本，预处理，分词，得到词典
raw_word_list = []
path = './negativeReviews'
idx=0
for filename in os.listdir(path):
    logging.debug(str(len(raw_word_list))+'loading: '+str(path)+'/'+filename)
    
    with open(path+'/'+filename,"r") as f:#filter
        idx+=1
        if idx >500:
            break
        for line in f:
            #clear special character:only chinese
            line = re.sub("[^\u4e00-\u9fff]", "", line)
            if len(line)>=2:
                logging.info(str(idx)+' loading text:'+line[:10]+'......')
                while '\n' in line:
                    line = line.replace('\n','')
                while ' ' in line:
                    line = line.replace(' ','')
                if len(line)>0: # 如果句子非空
                    raw_words = list(jieba.cut(line,cut_all=False))
                    raw_filterStopword = []
                    for w in raw_words:
                        if w not in stop_words:
                            raw_word_list.append(w)
words = raw_word_list
cWords = collections.Counter(words)

print(cWords.most_common(50))
