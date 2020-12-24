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

sys.path.append('../../../content_process_tool/jieba_zn/')
import jieba
from jieba import posseg as pseg
import jieba.analyse

from io import open
import random


#Step 1.1: load word embeddings data

Json_file = open("./znWord2Vec300.txt","r")
Json_file = json.load(Json_file)
wordsList = list(np.load('idxWord.npy'))
print(wordsList[0])
wordVectors=[]
wordVectors = list(np.load('idxWordVectors.npy'))

print("向量 -> "+str(wordVectors[wordsList.index("資料")]))


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

