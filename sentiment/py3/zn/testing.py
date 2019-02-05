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
logging.info("向量 -> "+str(wordVectors[wordsList.index("資料")]))


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
#step2 is for preprocessing of training and testing data in the data folder called 'positiveReviews' and 'negativeReviews'

#Step 2.1: build embeddings structure of the sentence.
#Could skip this step if the result was got.


positiveFiles = ['positiveReviews/' + f for f in listdir('positiveReviews/') if isfile(join('positiveReviews/', f))]
negativeFiles = ['negativeReviews/' + f for f in listdir('negativeReviews/') if isfile(join('negativeReviews/', f))]
numWords = []
for pf in positiveFiles:
    with open(pf, "r", encoding='utf-8') as f:
        
        linearray = f.readlines()
        line= "".join(linearray)
        counter = len(jieba.lcut(line))
        
        numWords.append(counter)
logging.info('Positive files finished')

for nf in negativeFiles:
    with open(nf, "r", encoding='utf-8') as f:
        linearray = f.readlines()
        line= "".join(linearray)
        counter = len(jieba.lcut(line))
        numWords.append(counter)
logging.info('Negative files finished')

numFiles = len(numWords)
logging.info('The total number of files is '+str(numFiles))
logging.info('The total number of words in the files is '+str(sum(numWords)))
logging.info('The average number of words in the files is '+str(sum(numWords)/len(numWords)))
maxSeqLength = 50


#Step 2.2: convert words of the sentence to vector and insert to structure.
#Removes punctuation, parentheses, question marks, etc., and leaves only alphanumeric characters.
def cleanSentences(content):
    #clear special character:only chinese
    content = re.sub("[^\u4e00-\u9fff]", "", content)
    logging.debug(content)
    return jieba.lcut(content)

ids = np.zeros((numFiles, maxSeqLength), dtype='int32')
fileCounter = 0
logging.info("positiveFiles length: "+str(len(positiveFiles)))
logging.info("negativeFiles length: "+str(len(negativeFiles)))
for pf in positiveFiles:
   logging.debug(pf)
   with open(pf, "r") as f:
       indexCounter = 0
       linearray = f.readlines()
       line= "".join(linearray)
       #cleanedLine = cleanSentences(line)
       #split = cleanedLine.split()
       logging.debug(line)
       split = cleanSentences(line)
       logging.debug(split)
       if len(split)<4:#ignoring the file if length don't fit to training
           continue
       
       for word in split:
           
               
           try:
               logging.debug(word)
               ids[fileCounter][indexCounter] = wordsList.index(word)
      
           except ValueError:
               pass
           
           indexCounter = indexCounter + 1
           if indexCounter >= maxSeqLength :#ignoring continue string if length don't fit to training.
               break
       logging.debug(ids[fileCounter])
       fileCounter = fileCounter + 1

for nf in negativeFiles:
   logging.debug(nf)
   with open(nf, "r") as f:
       indexCounter = 0
       linearray = f.readlines()
       line= "".join(linearray)
       split = cleanSentences(line)
       logging.debug(split)
       if len(split)<4:#ignoring the file if length don't fit to training
           continue
       
       for word in split:
           try:
               logging.debug(word)
               ids[fileCounter][indexCounter] = wordsList.index(word)
      
           except ValueError:
               pass
           indexCounter = indexCounter + 1
           if indexCounter >= maxSeqLength:#ignoring continue string it if length don't fit to training.
               break
   
       logging.debug(ids[fileCounter])
       fileCounter = fileCounter + 1
np.save('idsMatrix', ids)
logging.info("np.save('idsMatrix', ids)")


'''


#Step3.1: loading sample
ids = np.load('idsMatrix.npy')


#Step3.2: defind config for training

batchSize = 100
lstmUnits = 64
numClasses = 2
iterations = 100001   
maxSeqLength = 50
numDimensions = 300

positiveFilesCount = len(listdir('positiveReviews/'))#5251
negativeFilesCount = len(listdir('negativeReviews/'))#4764
testingFilesCount = int((positiveFilesCount+negativeFilesCount)*0.1)#1000

def getTrainBatch():
    testRange = testingFilesCount/2
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        if (i % 2 == 0):#positive
            num = randint(1,int(positiveFilesCount-testRange))
            labels.append([1,0])
        else:#negative
            num = randint(int(positiveFilesCount+testRange),int(positiveFilesCount+negativeFilesCount-1))
            labels.append([0,1])
        logging.debug(str(num))
        arr[i] = ids[num-1:num]


    return arr, labels

def getTestBatch(type='default'):
    testRange = testingFilesCount/2
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        num = randint(int(positiveFilesCount-testRange),int(positiveFilesCount+testRange))
        if (num <= positiveFilesCount):
            labels.append([1,0])
        else:
            labels.append([0,1])
        arr[i] = ids[num-1:num]
    return arr, labels
    


tf.reset_default_graph()




with tf.name_scope('Embeddings'):
    labels = tf.placeholder(tf.float32, [batchSize, numClasses])
    input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])

    data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]),dtype=tf.float32)
    data = tf.nn.embedding_lookup(wordVectors,input_data)

with tf.name_scope('rnn'):
    #lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
    lstmCell = tf.nn.rnn_cell.LSTMCell(name='basic_lstm_cell',num_units=lstmUnits)
    lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell,input_keep_prob=1.0, output_keep_prob=0.75)

    data = tf.cast(data,tf.float32)
    rnn_out, state = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)
    tf.summary.histogram('rnn_out', rnn_out)

with tf.name_scope('hidden1'):
    value = tf.transpose(rnn_out, [1, 0, 2])
    rnn_last_output = tf.gather(value, int(value.get_shape()[0]) - 1)
    with tf.variable_scope('weight'):
        #weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]),name="weight")
	    weight = tf.Variable(tf.random_normal([lstmUnits, numClasses]),name="weight")
	    tf.summary.histogram('weight', weight)
        
    with tf.variable_scope('bias'):
	    #bias = tf.Variable(tf.constant(0.1, shape=[numClasses]),name="bias")
	    bias = tf.Variable(tf.random_normal([numClasses]),name="bias")
	    tf.summary.histogram('bias', bias)

    #prediction = tf.nn.softmax(tf.add(tf.matmul(rnn_last_output, weight),bias))
    #prediction = tf.add(tf.matmul(rnn_last_output, weight),bias)
    prediction = tf.nn.tanh(tf.add(tf.matmul(rnn_last_output, weight),bias))
    #prediction = tf.nn.softmax(tf.add(tf.matmul(rnn_last_output, weight),bias))
    #prediction = tf.nn.relu(tf.add(tf.matmul(rnn_last_output, weight),bias))

    
'''

with tf.name_scope('hidden1'):
    value = tf.transpose(rnn_out, [1, 0, 2])
    rnn_last_output = tf.gather(value, int(value.get_shape()[0]) - 1)
    with tf.variable_scope('hidden1'):
        weight = tf.Variable(tf.truncated_normal([64, 32]),name="weight")
        bias = tf.Variable(tf.constant(0.1, shape=[32]),name="bias")
    #prediction = tf.add(tf.matmul(rnn_last_output, weight),bias)
    prediction_hidden1 = tf.nn.tanh(tf.add(tf.matmul(rnn_last_output, weight),bias))
    #prediction = tf.nn.softmax(tf.add(tf.matmul(rnn_last_output, weight),bias))
    #prediction = tf.nn.relu(tf.add(tf.matmul(rnn_last_output, weight),bias))
    tf.summary.histogram('hid_out_for_prediction_hidden1', prediction_hidden1)


with tf.name_scope('hidden2'):
    with tf.variable_scope('hidden2'):
        weight = tf.Variable(tf.truncated_normal([32, 2]),name="weight")
        bias = tf.Variable(tf.constant(0.1, shape=[2]),name="bias")
    prediction = tf.nn.tanh(tf.add(tf.matmul(prediction_hidden1, weight),bias))
    tf.summary.histogram('hid_out_for_prediction_hidden2', prediction)
'''

with tf.name_scope('accuracy'):
    correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
    accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

with tf.name_scope('loss'):
    #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=labels))
    

with tf.name_scope('optimize'):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.03).minimize(loss)#learning_rate inital value is 0.001
    #optimizer = tf.train.AdagradOptimizer(learning_rate=0.03).minimize(loss)
    #optimizer = tf.train.MonentumOptimizer(learning_rate=0.03,monentum=0.9).minimize(loss)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.35).minimize(loss)
sess = tf.Session()

'''
#Step3.3: defind config for Tensorboard
tf.summary.scalar('Loss', loss)
tf.summary.scalar('Accuracy', accuracy)
merged = tf.summary.merge_all()
logdir = "tensorboard/H5_tanh_LR0001_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir, graph =sess.graph)
'''

'''
#Step4.0: inital config of training
sess = tf.InteractiveSession()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

print("training process start")
#Step4.1: training
for i in range(iterations):
   #Next Batch of reviews
   nextBatch, nextBatchLabels = getTrainBatch()
   logging.debug(nextBatch)
   _,stepLoss = sess.run([merged,loss], {input_data: nextBatch, labels: nextBatchLabels})

   #Write summary to Tensorboard
   if (i % 50 == 0):
       
       #getTestBatch
       nextBatch, nextBatchLabels = getTestBatch()

       summary,stepLoss,stepAccuracy = sess.run([merged,loss,accuracy], {input_data: nextBatch, labels: nextBatchLabels}) 
       writer.add_summary(summary, i)
       print(str(i),"Accuracy:",stepAccuracy* 100,"Loss: ",stepLoss)

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

'''
#Step5.1: testing the model
iterations = 10
for i in range(iterations):
    nextBatch, nextBatchLabels = getTestBatch();
    print("Accuracy for this batch:", (sess.run(accuracy, {input_data: nextBatch, labels: nextBatchLabels})) * 100)




#Step5.2: testing the model by string
def testingSpeech(source):
    #clear special character:only chinese
    content = re.sub("[^\u4e00-\u9fff]", "", source)
    logging.debug(source)
    source_list = jieba.lcut(source)
    Sentence = np.zeros((maxSeqLength), dtype='int32')
    idx = 0
    for i in source_list:
        try:
            Sentence[idx] = wordsList.index(i)
            print(i)
        except:
            Sentence[idx] =0
        idx+=1
        if idx>=maxSeqLength:
            break

    input_data_Sentence = np.zeros((batchSize,maxSeqLength))
    input_data_Sentence[0] = Sentence
    prediction_answer = sess.run(prediction,feed_dict={input_data: input_data_Sentence})
    prediction_answer = prediction_answer[0]
    print(source)
    print(prediction_answer)
    if prediction_answer[0]>prediction_answer[1]:
        print("good",prediction_answer[0]-prediction_answer[1])
    else:
        print("bad",prediction_answer[1]-prediction_answer[0])
    print("-"*10)


testingSpeech("漏洞")
testingSpeech("妳的論述並不合理")
testingSpeech("我不同意你的看法")
testingSpeech("我認為你的推論很不正確除非我有看到更多的證據來判斷")
testingSpeech("我同意你的看法")


