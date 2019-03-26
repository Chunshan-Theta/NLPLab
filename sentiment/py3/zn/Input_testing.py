#-*- coding:utf-8 -*-
from os import environ
environ['TF_CPP_MIN_LOG_LEVEL']='2' #silence waring logs
import numpy as np
import tensorflow as tf
from os import listdir
from os.path import isfile, join
import re
import logging
logging.basicConfig(level=logging.INFO)
from random import randint
import datetime
import json
import sys
'''
import jieba
import jieba.posseg as pseg
import jieba.analyse
'''

sys.path.append('../../../jieba_zn/')
import jieba
from jieba import posseg as pseg
import jieba.analyse

from io import open
import random


#Step 1.1: load word embeddings data
'''
wordsList = np.load('wordsList.npy')
print('Loaded the word list!')
wordsList = wordsList.tolist() #Originally loaded as numpy array
wordsList = [word.decode('UTF-8') for word in wordsList] #Encode words as UTF-8
wordVectors = np.load('wordVectors.npy')
print ('Loaded the word vectors!')

logging.info(str(len(wordsList)))
assert type(wordVectors.shape) is tuple
'''


Json_file = open("./znWord2Vec300.txt","r")
Json_file = json.load(Json_file)
wordsList = list(np.load('idxWord.npy'))
print(wordsList[0])
wordVectors=[]
for w in wordsList:
    wordVectors.append(Json_file[w])
wordVectors = np.asarray(wordVectors)

finWord = int(len(wordsList))
assert type(wordVectors.shape) is tuple

'''
#using:
baseballIndex = wordsList.index('baseball')
print(wordVectors[baseballIndex])
'''
logging.debug("向量 -> "+str(wordVectors[wordsList.index("資料")]))


'''
#Step 2.01: Example for build embeddings structure of the sentence.
#现在我们有了向量，我们的第一步就是输入一个句子，然后构造它的向量表示。假设我们现在的输入句子是 “I thought the movie was incredible and inspiring”。为了得到词向量，我们可以使用 TensorFlow 的嵌入函数。这个函数有两个参数，一个是嵌入矩阵（在我们的情况下是词向量矩阵），另一个是每个词对应的索引。接下来，让我们通过一个具体的例子来说明一下。



maxSeqLength = 10 #Maximum length of sentence
numDimensions = 300 #Dimensions for each word vector
firstSentence = np.zeros((maxSeqLength), dtype='int32')
firstSentence[0] = wordsList.index("i")
firstSentence[1] = wordsList.index("thought")
firstSentence[2] = wordsList.index("the")
firstSentence[3] = wordsList.index("movie")
firstSentence[4] = wordsList.index("was")
firstSentence[5] = wordsList.index("incredible")
firstSentence[6] = wordsList.index("and")
firstSentence[7] = wordsList.index("inspiring")
#firstSentence[8] and firstSentence[9] are going to be 0

logging.info("\nfirstSentence.shape: "+str(firstSentence.shape))
logging.info("\nfirstSentence: "+str(firstSentence)) #Shows the row index for each word

with tf.Session() as sess:
    print(tf.nn.embedding_lookup(wordVectors,firstSentence).eval().shape)
'''

def SentencesCuter(source):
    source = source.replace(" ", "") #remove space
    #clear special character:only chinese
    source_list = jieba.lcut(source, cut_all=True)
    source = "".join(source_list)
    source = re.sub("[^\u4e00-\u9fff]", "", source)
    words = pseg.cut(source)
    re_lcut=[]
    allowedtype=["n","v","vd","vn","ns","a","d","ad","an","x"]
    for word, flag in words:
        
        if flag in allowedtype:
            re_lcut.append(word)
        else:
            pass

    
    return re_lcut
'''
#step2 is for preprocessing of training and testing data in the data folder called 'positiveReviews' and 'negativeReviews'

#Step 2.1: build embeddings structure of the sentence.
#Could skip this step if the result was got.


positiveFiles = ['positiveReviews/' + f for f in listdir('positiveReviews/') if isfile(join('positiveReviews/', f))]
negativeFiles = ['negativeReviews/' + f for f in listdir('negativeReviews/') if isfile(join('negativeReviews/', f))]
numWords = []

for pf in positiveFiles:
    
    with open(pf, "r", encoding='utf-8') as f:
        logging.info(str(pf))
        linearray = f.readlines()
        line= "".join(linearray)
        counter = len(SentencesCuter(line))
        
        numWords.append(counter)
logging.info('Positive files finished')

for nf in negativeFiles:
    
    with open(nf, "r", encoding='utf-8') as f:
        logging.info(str(nf))
        linearray = f.readlines()
        line= "".join(linearray)
        counter = len(SentencesCuter(line))
        numWords.append(counter)
logging.info('Negative files finished')

numFiles = len(numWords)
print('The total number of files is ',numFiles)
print('The total number of words in the files is ',sum(numWords))
print('The average number of words in the files is ',sum(numWords)/len(numWords))
maxSeqLength = 35


#Step 2.2: convert words of the sentence to vector and insert to structure.


ids = np.zeros((numFiles, maxSeqLength), dtype='float32')
fileCounter = 0
len_positiveFiles =len(positiveFiles)
len_negativeFiles =len(negativeFiles)
num_of_file = len_negativeFiles+len_positiveFiles
logging.info("positiveFiles length: "+str(len(positiveFiles)))
logging.info("negativeFiles length: "+str(len(negativeFiles)))
for pf in positiveFiles:
   logging.debug(pf)
   with open(pf, "r") as f:
       indexCounter = 0
       linearray = f.readlines()
       line= "".join(linearray)
       
       logging.debug(line)
       split = SentencesCuter(line)
       logging.debug(split)
       
       for word in split:
           
           
           try:
               
               ids[fileCounter][indexCounter] = wordsList.index(word)
               
           except ValueError:
               logging.debug("ignored: ",word)
               #ids[fileCounter][indexCounter] = 0
           
           indexCounter = indexCounter + 1
           if indexCounter >= maxSeqLength :#ignoring continue string if length don't fit to training.
               break

             
       print(fileCounter,'/',num_of_file)
       fileCounter = fileCounter + 1
       
for nf in negativeFiles:
   logging.debug(nf)
   with open(nf, "r") as f:
       indexCounter = 0
       linearray = f.readlines()
       line= "".join(linearray)
       

       logging.debug(line)
       split = SentencesCuter(line)
       logging.debug(split)
       

       for word in split:
           
           
           try:
               
               ids[fileCounter][indexCounter] = wordsList.index(word)
               
           except ValueError:
               logging.debug("ignored: ",word)
               #ids[fileCounter][indexCounter] = 0
           
           indexCounter = indexCounter + 1
           if indexCounter >= maxSeqLength :#ignoring continue string if length don't fit to training.
               break
       
            
       print(fileCounter,'/',num_of_file)

       fileCounter = fileCounter + 1
      
np.save('idsMatrix', ids)
logging.info("np.save('idsMatrix', ids)")


'''

#Step3.1: loading sample
ids = np.load('idsMatrix.npy')


#Step3.2: defind config for training

batchSize = 6
lstmUnits = 64
numClasses = 2
iterations = 100000
maxSeqLength = 35
numDimensions = 300

positiveFilesCount = len(listdir('positiveReviews/'))#5251
negativeFilesCount = len(listdir('negativeReviews/'))#4764
testingFilesCount = int((positiveFilesCount+negativeFilesCount)*0.3)#1000
halfTestRange = int(testingFilesCount/2)
print(positiveFilesCount,negativeFilesCount,testingFilesCount)
def getTrainBatch1():
    
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        if (i % 2 == 0):
            #pick the sample from the positive snetence set
            RandomStart = 1
            RandomStop = int(positiveFilesCount-halfTestRange)
            

            num = randint(RandomStart,RandomStop) 
            #if the sample sentence is meanless, re-random it. 
            while 3> np.count_nonzero(ids[num-1:num][0]):num = randint(RandomStart,RandomStop)
                
            labels.append([1,0])
        else:
            #pick the sample from the negative sentence set
            RandomStart = int(positiveFilesCount+halfTestRange)
            RandomStop = int(positiveFilesCount+negativeFilesCount-1)
            
            num = randint(RandomStart,RandomStop)
            #if the sample sentence is meanless, re-random it.            
            while 3> np.count_nonzero(ids[num-1:num][0]):num = randint(RandomStart,RandomStop)              
            labels.append([0,1])
        arr[i] = ids[num-1:num]
        arr[i] = random.sample(list(arr[i]), len(arr[i]))
    return arr, labels


def getTrainBatch2():
    
    input_data_Sentence=[]
    positiveFiles = ['positiveReviews/' + f for f in listdir('positiveReviews/') if isfile(join('positiveReviews/', f))]
    negativeFiles = ['negativeReviews/' + f for f in listdir('negativeReviews/') if isfile(join('negativeReviews/', f))]
    labels=[]
    
    while len(input_data_Sentence)!=(batchSize/2):
        num = randint(0,int(positiveFilesCount-halfTestRange)-1)
        with open(positiveFiles[num], "r", encoding='utf-8') as f:
            vaildword=0
            linearray = f.readlines()
            source= "".join(linearray)
            source_list = SentencesCuter(source)
            Sentence= np.zeros((maxSeqLength), dtype='int32')
            
            idx = 0
            for i in source_list:
                try:
                    Sentence[idx] = float(wordsList.index(i))
                    vaildword+=1
                except:
                    pass#Sentence[idx] =0
                idx+=1

                #break the process if the sentence too long
                if idx>=maxSeqLength:break    
            #print(source_list)
            if vaildword>3:
                labels.append([1,0])
                input_data_Sentence.append(list(Sentence))
            
    
    while len(input_data_Sentence)!=batchSize:
        num = randint(int(halfTestRange)-1,int(negativeFilesCount)-1)
        with open(negativeFiles[num], "r", encoding='utf-8') as f:
            vaildword=0
            linearray = f.readlines()
            source= "".join(linearray)
            source_list = SentencesCuter(source)
            Sentence= np.zeros((maxSeqLength), dtype='int32')
            
            idx = 0
            for i in source_list:
                try:
                    Sentence[idx] = float(wordsList.index(i))
                    vaildword+=1
                except:
                    pass#Sentence[idx] =0
                idx+=1

                #break the process if the sentence too long
                if idx>=maxSeqLength:break    
            #print(source_list)
            if vaildword>3:
                labels.append([0,1])
                input_data_Sentence.append(list(Sentence))
            
    arr = np.asarray(input_data_Sentence[:]).astype(float)
    

    return arr, labels  
  
def getTrainBatch():
    
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        if (i % 2 == 0):
            #pick the sample from the positive snetence set
            RandomStart = 1
            RandomStop = int(positiveFilesCount-halfTestRange)
            

            num = randint(RandomStart,RandomStop) 
            #if the sample sentence is meanless, re-random it. 
            while 3> np.count_nonzero(ids[num-1:num][0]):num = randint(RandomStart,RandomStop)
                
            labels.append([1,0])
        else:
            #pick the sample from the negative sentence set
            RandomStart = int(positiveFilesCount+halfTestRange)
            RandomStop = int(positiveFilesCount+negativeFilesCount-1)
            
            num = randint(RandomStart,RandomStop)
            #if the sample sentence is meanless, re-random it.            
            while 3> np.count_nonzero(ids[num-1:num][0]):num = randint(RandomStart,RandomStop)              
            labels.append([0,1])
        arr[i] = ids[num-1:num]
        arr[i] = random.sample(list(arr[i]), len(arr[i]))
    

    return arr, labels

def getTestBatch():
    

    input_data_Sentence=[]
    positiveFiles = ['positiveReviews/' + f for f in listdir('positiveReviews/') if isfile(join('positiveReviews/', f))]
    negativeFiles = ['negativeReviews/' + f for f in listdir('negativeReviews/') if isfile(join('negativeReviews/', f))]
    labels=[]
    
    while (batchSize/2)>=len(input_data_Sentence):
        num = randint(int(positiveFilesCount-halfTestRange),int(positiveFilesCount)-1)
        with open(positiveFiles[num], "r", encoding='utf-8') as f:
            vaildword=0
            linearray = f.readlines()
            source= "".join(linearray)
            source_list = SentencesCuter(source)
            Sentence= np.zeros((maxSeqLength), dtype='int32')
            
            idx = 0
            for i in source_list:
                try:
                    Sentence[idx] = float(wordsList.index(i))
                    vaildword+=1
                except:
                    pass#Sentence[idx] =0
                idx+=1

                #break the process if the sentence too long
                if idx>=maxSeqLength:break    
            #print(source_list)
            if vaildword>3:
                labels.append([1,0])
                input_data_Sentence.append(list(Sentence))
            
    
    while batchSize>len(input_data_Sentence):
        num = randint(0,int(halfTestRange)-1)
        with open(negativeFiles[num], "r", encoding='utf-8') as f:
            vaildword=0
            linearray = f.readlines()
            source= "".join(linearray)
            source_list = SentencesCuter(source)
            Sentence= np.zeros((maxSeqLength), dtype='int32')
            
            idx = 0
            for i in source_list:
                try:
                    Sentence[idx] = float(wordsList.index(i))
                    vaildword+=1
                except:
                    pass#Sentence[idx] =0
                idx+=1

                #break the process if the sentence too long
                if idx>=maxSeqLength:break    
            #print(source_list)
            if vaildword>3:
                labels.append([0,1])
                input_data_Sentence.append(list(Sentence))
            
    arr = np.asarray(input_data_Sentence[:]).astype(float)
    

    return arr, labels

nextBatch, nextBatchLabels = getTrainBatch()
print(nextBatch)
for i in nextBatch :
    text=""
    for t in i:
        text+=wordsList[int(t)]+"|" if int(t)!=0 else ""
    print(text)
print('-'*20)
nextBatch, nextBatchLabels = getTestBatch()
print(nextBatch)
for i in nextBatch :
    text=""
    for t in i:
        text+=wordsList[int(t)]+"|" if int(t)!=0 else ""
    print(text)

