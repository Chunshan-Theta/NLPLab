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
sys.path.append('../../jieba_zn/')
import jieba
import jieba.posseg as pseg
import jieba.analyse
from io import open



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


Json_file = open("./znWord2Vec.txt","r")
Json_file = json.load(Json_file)
wordsList = [word for word in Json_file]
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
logging.debug("資料 vec: "+str(wordVectors[wordsList.index(unicode("資料","utf-8"))]))


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



'''
#Step 2.1: build embeddings structure of the sentence.
#Could skip this step if the result was got.


positiveFiles = ['positiveReviews/' + f for f in listdir('positiveReviews/') if isfile(join('positiveReviews/', f))]
negativeFiles = ['negativeReviews/' + f for f in listdir('negativeReviews/') if isfile(join('negativeReviews/', f))]
numWords = []
for pf in positiveFiles:
    with open(pf, "r", encoding='utf-8') as f:
        line=f.readline()
        #counter = len(line.split())
        counter = len(jieba.lcut(line))
        numWords.append(counter)
logging.info('Positive files finished')

for nf in negativeFiles:
    with open(nf, "r", encoding='utf-8') as f:
        line=f.readline()
        #counter = len(line.split())
        counter = len(jieba.lcut(line))
        numWords.append(counter)
logging.info('Negative files finished')

numFiles = len(numWords)
logging.info('The total number of files is', numFiles)
logging.info('The total number of words in the files is', sum(numWords))
logging.info('The average number of words in the files is', sum(numWords)/len(numWords))
maxSeqLength = 35


#Step 2.2: convert words of the sentence to vector and insert to structure.
#Removes punctuation, parentheses, question marks, etc., and leaves only alphanumeric characters.
def cleanSentences(content):
    content = content.replace(" ", "") #remove space
    content = re.sub("[A-Za-z0-9]+", "", content) # remove English and number
    content = re.sub("[/.~?=,]+", "", content) # remove Any character symbols
    return jieba.lcut(content)

ids = np.zeros((numFiles, maxSeqLength), dtype='int32')
fileCounter = 0
logging.info("positiveFiles length: "+str(len(positiveFiles)))
logging.info("negativeFiles length: "+str(len(negativeFiles)))
for pf in positiveFiles:
   with open(pf, "r") as f:
       indexCounter = 0
       line=f.readline()
       #cleanedLine = cleanSentences(line)
       #split = cleanedLine.split()
       split = cleanSentences(line)
       for word in split:
           try:
               if len(word)<=1:
                   ids[fileCounter][indexCounter] = finWord-1 #ignore mean-less words
               else:
                   logging.info(word)
                   ids[fileCounter][indexCounter] = wordsList.index(word)
           except ValueError:
               ids[fileCounter][indexCounter] = finWord-1 #Vector for unkown words
           indexCounter = indexCounter + 1
           if indexCounter >= maxSeqLength:
               break
       fileCounter = fileCounter + 1

for nf in negativeFiles:
   with open(nf, "r") as f:
       indexCounter = 0
       line=f.readline()
       #cleanedLine = cleanSentences(line)
       #split = cleanedLine.split()
       split = cleanSentences(line)
       for word in split:
           try:
               ids[fileCounter][indexCounter] = wordsList.index(word)
               if len(word)<=1:
                   ids[fileCounter][indexCounter] = finWord-1 #ignore mean-less words
           except ValueError:
               ids[fileCounter][indexCounter] = finWord-1 #Vector for unkown words
           indexCounter = indexCounter + 1
           if indexCounter >= maxSeqLength:
               break
       fileCounter = fileCounter + 1
np.save('idsMatrix', ids)
logging.info("np.save('idsMatrix', ids)")


'''


#Step3.1: loading sample
ids = np.load('idsMatrix.npy')


#Step3.2: defind config for training

batchSize = 24
lstmUnits = 64
numClasses = 2
iterations = 100000
maxSeqLength = 35
numDimensions = 2

positiveFilesCount = 5251
negativeFilesCount = 4764
testingFilesCount = 1000

def getTrainBatch():
    testRange = testingFilesCount/2
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        if (i % 2 == 0):#positive
            num = randint(1,positiveFilesCount-testRange)
            labels.append([1,0])
        else:#negative
            num = randint(positiveFilesCount+testRange,positiveFilesCount+negativeFilesCount-1)
            labels.append([0,1])

        arr[i] = ids[num-1:num]

    return arr, labels

def getTestBatch():
    testRange = testingFilesCount/2
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        num = randint(positiveFilesCount-testRange,positiveFilesCount+testRange)
        if (num <= positiveFilesCount):
            labels.append([1,0])
        else:
            labels.append([0,1])
        arr[i] = ids[num-1:num]
    return arr, labels


sess = tf.Session()
tf.reset_default_graph()

labels = tf.placeholder(tf.float32, [batchSize, numClasses])
input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])

data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]),dtype=tf.float32)
data = tf.nn.embedding_lookup(wordVectors,input_data)

lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
data = tf.cast(data,tf.float32)
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = (tf.matmul(last, weight) + bias)

correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
optimizer = tf.train.AdamOptimizer().minimize(loss)


#Step3.3: defind config for Tensorboard
tf.summary.scalar('Loss', loss)
tf.summary.scalar('Accuracy', accuracy)
merged = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)


'''
#Step4.0: inital config of training
sess = tf.InteractiveSession()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

#Step4.1: training
for i in range(iterations):
   #Next Batch of reviews
   nextBatch, nextBatchLabels = getTrainBatch();
   sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels})

   #Write summary to Tensorboard
   if (i % 50 == 0):
       summary = sess.run(merged, {input_data: nextBatch, labels: nextBatchLabels})
       writer.add_summary(summary, i)

       #Accuracy
       nextBatch, nextBatchLabels = getTestBatch();
       print("Accuracy for this batch:", (sess.run(accuracy, {input_data: nextBatch, labels: nextBatchLabels})) * 100)

   #Save the network every 10,000 training iterations
   if (i % 10000 == 0 and i != 0):
       save_path = saver.save(sess, "models/pretrained_lstm.ckpt", global_step=i)
       print("saved to %s" % save_path)

writer.close()
'''

#Step5.0: loading model

sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, tf.train.latest_checkpoint('models'))


'''
#Step5.1: testing the model
iterations = 10
for i in range(iterations):
    nextBatch, nextBatchLabels = getTestBatch();
    print("Accuracy for this batch:", (sess.run(accuracy, {input_data: nextBatch, labels: nextBatchLabels})) * 100)
'''



#Step5.2: testing the model by string
def testingSpeech(source):
    source = source.replace(" ", "") #remove space
    source = re.sub("[A-Za-z0-9]+", "", source) # remove English and number
    source = re.sub("[/.~?=,]+", "", source) # remove Any character symbols
    source_list = jieba.lcut(source)
    Sentence = np.zeros((maxSeqLength), dtype='int32')
    idx = 0
    for i in source_list:
        try:
            if len(i)<=1:
                Sentence[idx] =finWord-1
            else:
                Sentence[idx] = wordsList.index(unicode(i,"utf-8"))
        except:
            Sentence[idx] =finWord-1
        idx+=1
        if idx>=maxSeqLength:
            break

    input_data_Sentence = np.zeros((24,35))
    input_data_Sentence[0] = Sentence
    prediction_answer = sess.run(prediction,feed_dict={input_data: input_data_Sentence})
    prediction_answer = prediction_answer[0]
    print(source)
    if prediction_answer[0]>prediction_answer[1]:
        print("good",prediction_answer[0]-prediction_answer[1])
    else:
        print("bad",prediction_answer[1]-prediction_answer[0])
    print("-"*10)

testingSpeech("[其他]  也有人們相對會提出：建造核廠的資金非常龐大，難道全都要由人民的血汗錢支出？說實在的，建造核廠所需的錢確實很多，但各位是否思考過，用火力發電，要燃燒石油煤碳與天然氣，現在的石油在數十年後就面臨枯竭的危機，那時購買原料的價錢就會變貴，因此，何不直接停止石化原料燃燒發電的方法，用費用較便宜的鈾料產生電力就好了呢？")
testingSpeech("[提出疑問]  是阿~這就是剛才貼的東西寫的")
testingSpeech("[其他]  看她漂亮啊")
testingSpeech("[提出疑問]  你能不能舉例說明")
testingSpeech("[提出疑問]  嗨嗨嗨嗨嗨嗨嗨嗨嗨嗨嗨嗨嗨嗨嗨嗨嗨嗨嗨嗨嗨嗨嗨嗨嗨嗨嗨嗨嗨嗨嗨嗨嗨嗨嗨嗨嗨嗨嗨嗨嗨嗨嗨")
testingSpeech("[提出疑問]  你能不能舉例說明關於核能對於環境破壞的可能性")
testingSpeech("[提出疑問]  你能不能舉例說明關於國營企業 核能對於環境破壞的可能性")
testingSpeech("[其他]  嗨")
testingSpeech("[提出論點或證據]  不過基本上 這樣國營企業要倒掉民營化是很難的")
testingSpeech("[其他]  你好友善 我旁邊感覺快吵起來惹哈哈哈")
testingSpeech("[表達同意或支持]  我應該是同意吧")
