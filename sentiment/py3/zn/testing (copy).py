import tensorflow as tf
import numpy as np
from os import listdir
from os.path import isfile, join
import re
import json
import random
from random import randint

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
    RandomStop =int(positiveFilesCount+testRange)
    for i in range(batchSize):
        num = randint(RandomStart,RandomStop)
        #if the sample sentence is meanless, re-random it.            
        while 3> np.count_nonzero(ids[num-1:num][0]):num = randint(RandomStart,RandomStop)

        #makeing testing Answer set.
        labels.append([1,0]) if num <= positiveFilesCount else labels.append([0,1])   
         
        arr[i] = ids[num-1:num]
    return arr, labels
    
tf.reset_default_graph()



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
            while 3> np.count_nonzero(ids[num-1:num][0]):num = randint(RandomStart,RandomStop)
                
            labels.append([1,0])     
        else:
            #pick the sample from the negative sentence set
            RandomStart = int(positiveFilesCount+testRange)
            RandomStop = int(positiveFilesCount+negativeFilesCount-1)
            
            num = randint(RandomStart,RandomStop)
            #if the sample sentence is meanless, re-random it.            
            while 3> np.count_nonzero(ids[num-1:num][0]):
                num = randint(RandomStart,RandomStop)              
            labels.append([0,1])
        arr[i] = ids[num-1:num]
        arr[i] = random.sample(list(arr[i]), len(arr[i]))
    
    return arr, labels


with tf.name_scope('Embeddings'):
    step = tf.placeholder(tf.int32)
    labels = tf.placeholder(tf.float32, [batchSize, numClasses])
    input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])
    
    data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]),dtype=tf.float32)
    data = tf.nn.embedding_lookup(wordVectors,input_data)
    learning_rate = tf.train.exponential_decay(learning_rate=0.03,
                                               global_step=step,
                                               decay_steps=50,
                                               decay_rate=0.9)

with tf.name_scope('rnn'):
    #lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
    lstmCell = tf.nn.rnn_cell.LSTMCell(name='basic_lstm_cell',num_units=lstmUnits)
    lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell,input_keep_prob=0.95, output_keep_prob=0.55)
    data = tf.cast(data,tf.float32)
    #value
    rnn_out,_ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)
    #tf.summary.histogram('rnn_out', rnn_out)

with tf.name_scope('hidden'):
    with tf.variable_scope('weight'):
        weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]),name="weight")
        tf.summary.histogram('weight', weight)
        
    with tf.variable_scope('bias'):
        bias = tf.Variable(tf.constant(0.1, shape=[numClasses]),name="bias")
        #bias = tf.Variable(tf.random_normal([numClasses]),name="bias")
        tf.summary.histogram('bias', bias)
    value = tf.transpose(rnn_out, [1, 0, 2])
    rnn_last_output = tf.gather(value, int(value.get_shape()[0]) - 1)
    
    logits = tf.matmul(rnn_last_output, weight)+bias
    
    prediction = logits
    #prediction = tf.nn.relu(tf.add(tf.matmul(rnn_last_output, weight),bias))
    #prediction = tf.nn.tanh(tf.add(tf.matmul(rnn_last_output, weight),bias))
    #prediction = tf.nn.softmax(tf.add(tf.matmul(rnn_last_output, weight),bias))

with tf.name_scope('accuracy'):
    correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
    accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

with tf.name_scope('loss'):
    #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=labels))
    

with tf.name_scope('optimizer'):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    #optimizer = tf.train.AdamOptimizer(learning_rate=0.03).minimize(loss)#learning_rate inital value is 0.001
    #optimizer = tf.train.AdagradOptimizer(learning_rate=0.03).minimize(loss)
    #optimizer = tf.train.MonentumOptimizer(learning_rate=0.03,monentum=0.9).minimize(loss)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.35).minimize(loss)
    



sess = tf.Session()

sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, tf.train.latest_checkpoint('models'))

iterations = 100
SumAccuracy = 0
for i in range(iterations):
    #nextBatch, nextBatchLabels = getTestBatch()
    nextBatch, nextBatchLabels = getTrainBatch();
    StepAccuracy=(sess.run(accuracy, {input_data: nextBatch, labels: nextBatchLabels})) * 100
    SumAccuracy+=StepAccuracy
    print("Accuracy for this batch:",StepAccuracy )
print(SumAccuracy/iterations)
    
    


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
    logging.debug(source)
    source_list = SentencesCuter(source)
    Sentence = np.zeros((maxSeqLength), dtype='int32')
    idx = 0
    validdwords=[]
    for i in source_list:
        try:
            Sentence[idx] = wordsList.index(i)
            logging.debug(i)
            validdwords.append(i)
        except:
            pass#Sentence[idx] =0
        idx+=1

        #break if sentence too long
        if idx>=maxSeqLength:break    

    input_data_Sentence = np.zeros((batchSize,maxSeqLength))
    input_data_Sentence[0] = Sentence
    prediction_result = sess.run(prediction,feed_dict={input_data: input_data_Sentence})
    prediction_answer = prediction_result[0][:]
    print(source,validdwords)
    if 4>len(validdwords): return [-1,'meanless']
    print(prediction_answer)
    #respond boolean base on prediction result.
    return [1,prediction_answer[0]] if prediction_answer[0]>prediction_answer[1] else [0,prediction_answer[1]]


#testingSpeech("妳的論述並不合理")
#for s in ["我同意你的看法","我認為你的推論很不正確除非我有看到更多的證據來判斷","我不同意你的看法","漏洞","妳的論述並不合理"]:
#    print(testingSpeech(s)[0],s)



'''
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
  "妳的說法還有一些漏洞但我大致上認同你的論述",
  "妳說的怪怪的但大致上你的論述還算合理"]

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

x = [
  "我認為你的推論很不正確除非我有看到更多的證據來判斷",
  "我沒辦法相信妳的推論除非妳有更多的證據可以證明",
  "你說的是錯誤的的確證據並沒有顯示出這樣的情況",
  "你說的聽起來很有道理但我們應該要拿出證據說話",
  "我能夠體會妳的想法但我認為其中有些問題"]

for s in p1+p2+p3:
    (testingSpeech(s)[0],s)

print("-"*20)
for s in n1+n2+n3:
    (testingSpeech(s)[0],s)

print("-"*20)
for s in x:
    (testingSpeech(s)[0],s)
'''
