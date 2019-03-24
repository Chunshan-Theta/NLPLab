import tensorflow as tf
import numpy as np
from os import listdir
from os.path import isfile, join
import re
import json
import random
from random import randint
import sys
sys.path.append('../../../jieba_zn/')
import jieba
from jieba import posseg as pseg
import jieba.analyse

#Step 1.1: load word embeddings dataa
Json_file = open("./znWord2Vec300.txt","r")
Json_file = json.load(Json_file)
wordsList = [word for word in Json_file]
print(wordsList[0])
wordVectors=[]
for w in wordsList:
    wordVectors.append(Json_file[w])
wordVectors = np.asarray(wordVectors)

finWord = int(len(wordsList))
assert type(wordVectors.shape) is tuple


#Step3.1: loading sample
ids = np.load('idsMatrix.npy')

#Step3.2: defind config for training
batchSize = 24
lstmUnits = 64
numClasses = 2
iterations = 100000
maxSeqLength = 35
numDimensions = 300

positiveFilesCount = len(listdir('positiveReviews/'))#5251
negativeFilesCount = len(listdir('negativeReviews/'))#4764
testingFilesCount = int((positiveFilesCount+negativeFilesCount)*0.3)#1000



def getTestBatch():
    testRange = testingFilesCount/2
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    RandomStart =int(positiveFilesCount-testRange)
    RandomStop =int(positiveFilesCount)#int(positiveFilesCount+testRange)
    for i in range(batchSize):
        num = randint(RandomStart,RandomStop)
        #if the sample sentence is meanless, re-random it.            
        while 1> np.count_nonzero(ids[num-1:num][0]):num = randint(RandomStart,RandomStop)

        #makeing testing Answer set.
        labels.append([1,0]) if num <= positiveFilesCount else labels.append([0,1])   
         
        arr[i] = ids[num-1:num]
    return arr, labels


def getTrainBatch():
    testRange = testingFilesCount/2
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        if (i % 2 == 0):
            #pick the sample from the positive snetence set
            RandomStart = 1
            RandomStop = int(positiveFilesCount-testRange)
            

            num = randint(RandomStart,RandomStop) 
            #if the sample sentence is meanless, re-random it. 
            while 1> np.count_nonzero(ids[num-1:num][0]):num = randint(RandomStart,RandomStop)
                
            labels.append([1,0])     
        else:
            #pick the sample from the negative sentence set
            RandomStart = int(positiveFilesCount+testRange)
            RandomStop = int(positiveFilesCount+negativeFilesCount-1)
            
            num = randint(RandomStart,RandomStop)
            #if the sample sentence is meanless, re-random it.            
            while 1> np.count_nonzero(ids[num-1:num][0]):
                num = randint(RandomStart,RandomStop)              
            labels.append([0,1])
        arr[i] = ids[num-1:num]
        arr[i] = random.sample(list(arr[i]), len(arr[i]))
    
    return arr, labels




    



# delete the current graph
tf.reset_default_graph()




sess = tf.Session()
# import the graph from the file
saver = tf.train.import_meta_graph('models/pretrained_lstm.ckpt-100000.meta')
# restore the saved vairable
saver.restore(sess, tf.train.latest_checkpoint('models'))
graph = tf.get_default_graph()


accuracy = tf.get_collection("accuracy")[0]
input_data = tf.get_collection("input_data")[0]
labels = tf.get_collection("labels")[0]
prediction = tf.get_collection("prediction")[0]
PredResult = tf.get_collection("PredResult")[0]
ModelAccuracy = 0
for i in range(10):
    # print the loaded variable
    nextBatch, nextBatchLabels = getTestBatch()
    StepAccuracy = sess.run(accuracy, {input_data: nextBatch, labels: nextBatchLabels})
    ModelAccuracy+=(StepAccuracy*100)
print('Models Accuracy: ',ModelAccuracy/10 ,'%')
#StepPredResult = sess.run([PredResult], {input_data: nextBatch})
#print(StepPredResult,nextBatchLabels)


def SentencesCuter(source):
    source = source.replace(" ", "") #remove space
    #clear special character:only chinese
    source_list = jieba.lcut(source, cut_all=False)
    source = "".join(source_list)
    source = re.sub("[^\u4e00-\u9fff]", "", source)
    words = pseg.cut(source)
    re_lcut=[]
    allowedtype=["n","v","vd","vn","ns","a","d","ad","an","x"]
    for word, flag in words:
        
        if flag in allowedtype:
            re_lcut.append(word)
        else:
            pass#print(word,flag)

    #print("".join(re_lcut))
    return re_lcut

#Step5.2: testing the model by string
def testingSpeech(source):


    source_list = SentencesCuter(source)
    Sentence = np.zeros((maxSeqLength), dtype='int32')
    idx = 0
    for i in source_list:
        try:
            Sentence[idx] = wordsList.index(i)
            
        except:
            pass#Sentence[idx] =0
        idx+=1

        #break the process if the sentence too long
        if idx>=maxSeqLength:break    
    print(source_list)


    #input_data_Sentence,input_data_Ans = getTestBatch()
    #input_data_Sentence[0] = Sentence

    input_data_Sentence=[Sentence for i in range(24)]
    p_result,p_value = sess.run([PredResult,prediction], {input_data: input_data_Sentence})
    
    
    prediction_rate = p_value[0][:]    
    prediction_answer = p_result[0]
    

    diff = prediction_rate[0]-prediction_rate[1]
    diff2 = sum([i[0]-i[1]for i in p_value])/24
    print(p_result,diff2) 
    print(prediction_answer,diff)
    print('----')
    return prediction_answer



'''
testingSpeech('房間糟透了 早餐也很難吃')
testingSpeech('房間很好 早餐好吃')
testingSpeech('服務員很親切 房間採光很好很舒適')
testingSpeech('服務員態度很糟糕 房間很小')
'''
positiveFiles = ['positiveReviews/' + f for f in listdir('positiveReviews/') if isfile(join('positiveReviews/', f))]
negativeFiles = ['negativeReviews/' + f for f in listdir('negativeReviews/') if isfile(join('negativeReviews/', f))]

for i in range(10):
    with open(positiveFiles[i], "r", encoding='utf-8') as f:
        
        linearray = f.readlines()
        line= "".join(linearray)
        print(line)
        testingSpeech(line)
for i in range(10):
    with open(negativeFiles[i], "r", encoding='utf-8') as f:
        
        linearray = f.readlines()
        line= "".join(linearray)
        print(line)
        testingSpeech(line)
