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
saver = tf.train.import_meta_graph('models_saved/pretrained_lstm.ckpt-100000.meta')
# restore the saved vairable
saver.restore(sess, tf.train.latest_checkpoint('models_saved'))
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


    input_data_Sentence,input_data_Ans = getTestBatch()
    input_data_Sentence[0] = Sentence

    
    p_result,p_value = sess.run([PredResult,prediction], {input_data: input_data_Sentence})
    
    
    prediction_rate = p_value[0][:]    
    prediction_answer = p_result[0]
    

    diff = prediction_rate[0]-prediction_rate[1] if prediction_rate[0]>prediction_rate[1] else prediction_rate[1]-prediction_rate[0]
    print(prediction_answer,diff)
    return prediction_answer




p1=["我同意",
  "你說的沒錯",
  "很好"]
p2=[
  "我認為我們的共識是正確的",
  "妳的論述很合理",
  "我認同你的論點",
  "你說的很好"]
p3=[
  "我認為你的推論很正確有很多證據可以證明你的論點",
  "我相信妳的推論聽起來很有道理",
  "你說的是正確的的確證據是顯示出這樣的情況",
  "妳的說法還有一些漏洞但我大致上認同你的論述"]

n1 = ["我不同意",
  "你說的並不正確",
  "我不同意你的看法"]
n2 = [
  "我們目前還沒有共識",
  "你說的部分有點問題",
  "妳的論述並不合理",
  "我沒辦法認同你的論點"]
n3 = [
  "我認為你的推論很不正確除非我有看到更多的證據來判斷",
  "我沒辦法相信妳的推論除非妳有更多的證據可以證明",
  "你說的是錯誤的的確證據並沒有顯示出這樣的情況",
  "你說的聽起來很有道理但我們應該要拿出證據說話",
  "我能夠體會妳的想法但我認為其中有些問題"]

n4 = [
  "唯一缺點就是燈很暗，最亮的竟然只有廁所燈...（白天不會）上去閣樓樓梯有點陡要小心，還有點小小殘忍的就是所有房型都禁止飲食，不過相信大家都會在台南的大街小巷吃得很滿足～",
  "轉程電梯 外加沒有無障礙設施 如果有兩間廁所 及浴室可以大間一點會更好 ",
  "  很糟糕的經驗，骯髒得很...而且地理位置也不好....",
  "  員工不太親切 臉很臭 床太軟 隔音差  在臉書上承諾我們的房間 到現場辦理入住卻給我們先看另一個房間，因那間真的不好，又問我們要不要加價換樓上房間，我就拿臉書給他看他當初承諾要給我的房間，為何不一開始就直接先給該給我們的房間呢，感覺不誠實!",
  "廁所比較小，但房間夠大，不計較了。"]
p4 = [
  "  溜滑梯房設計很好   孩子很喜歡 民宿老闆  老闆娘很友善👍",
  "夜景很美、地點佳交通方便、價格實在、浴室很乾淨",
  "  喜歡他在商圈附近，地標明顯。",
  "房間大，採光不錯",
  "空間寬敞 風景不錯"]

correctCount=0
for i in p1+p2+p3:
    ans =testingSpeech(i)
    if ans ==0 :correctCount+=1
print(correctCount/12*100)
correctCount=0
for i in n1+n2+n3:
    ans =testingSpeech(i)
    if ans ==1 :correctCount+=1
print(correctCount/12*100)

