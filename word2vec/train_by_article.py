# -*- coding:utf-8 -*-
#!usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import random
import sys
sys.path.append('../jieba_zn/')
import jieba
import numpy as np
from six.moves import xrange
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import datetime
import re
import logging
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import json
logging.basicConfig(level=logging.INFO)

if not os.path.exists("TB"):
    logging.info("create a folder: TB")
    os.makedirs("TB")

# Step 1: Download the data.
# Read the data into a list of strings.

def read_data():
    """
    对要训练的文本进行处理，最后把文本的内容的所有词放在一个列表中
    """
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

    path = './TextForTrain'

    for filename in os.listdir(path):
        logging.debug(str(len(raw_word_list))+'loading: '+str(path)+'/'+filename)
        with open(path+'/'+filename,"r") as f:#filter
            line = re.sub("[A-Za-z0-9]", "", f.readline())
            while line:#filter
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
                    #raw_word_list.extend(raw_words)
                line = re.sub("[A-Za-z0-9]", "", f.readline())
    return raw_word_list

#step 1:读取文件中的内容组成一个列表
logging.info("Begin loading step.");
words = read_data()
print('Data size', len(words))
logging.info("Begin loading completed.");


# Step 2: Build the dictionary and replace rare words with UNKNOWWORD token.
useValue = 0.8 #max is 1
vocabulary_size = int(len(collections.Counter(words))*useValue) #55000



assert useValue <= 1
print("count of dict:",len(collections.Counter(words)),"use :",vocabulary_size)
def build_dataset(words):
    count = [['UNKNOWWORD', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    print("count",len(count))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unknowword_count = 0
    for word in words:
        #if word.isdigit() or len(word)<2: #ignore the word that length is one or just combine with digit.
        if word.isdigit(): #ignore the word that just combine with digit.
            index = 0
            unknowword_count += 1
        elif word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unknowword_count += 1
        data.append(index)
    count[0][1] = unknowword_count
    #                               No.                  words
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words)
#删除words节省内存
del words

logging.info('Most common words (+UNKNOWWORD)'+str(count[:5]))
logging.info('Sample data'+str(data[:10])+str([reverse_dictionary[i] for i in data[:10]]))

data_index = 0


# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    #print("num_skips",num_skips,"skip_window",skip_window)
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels
'''
#调用generate_batch函数简单测试一下功能
batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
    #     14142     鬥破                           ->   7836          蒼穹
    print(batch[i], reverse_dictionary[batch[i]],'->', labels[i, 0], reverse_dictionary[labels[i, 0]])
'''

def showDetail(T_name,T,detail=0):
    logging.debug(str('*'*50))
    logging.debug("name:"+str(T_name))
    logging.debug("type:"+str(type(T)))
    try:
        logging.debug("len:"+str(len(T)))
    except:
        pass
    try:
        logging.debug("ndarray.shape:"+str(T.shape))
    except:
        pass
    if(detail):
        logging.debug(str(T))





# Step 4: Build and train a skip-gram model.
batch_size = 48
embedding_size = 600
skip_window = 1    # 2*skip_window >= num_skips
num_skips = 2       # = batch_size/n
num_sampled = 48    # Number of negative examples to sample.

assert batch_size%num_skips==0
assert 2*skip_window >= num_skips





starttime = datetime.datetime.now()

valid_word=[]
valid_word = ["貢獻","助於","錯誤","損失","遺憾"]



valid_size = len(valid_word)
'''
print(E_point-valid_size,E_point)

for i in xrange(E_point-valid_size,E_point):
    valid_word.append(reverse_dictionary[i].encode('utf-8'))
'''
#showDetail("valid_word",valid_word)
#print('valid_word')
#print(valid_word)

valid_examples =[dictionary[li] for li in valid_word]
graph = tf.Graph()
with graph.as_default():
    # Input data.
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size],name='train_inputs')
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1],name='train_labels')
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # Ops and variables pinned to the CPU because of missing GPU implementation
    with tf.device('/cpu:0'):
        # Look up embeddings for inputs.
        with tf.name_scope('Embeddings'):
            embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0),name='Embeddings')
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)
            tf.summary.histogram(name ='Embeddings', values = embeddings)

        # Construct the variables for the NCE loss
        with tf.name_scope('Weights'):
            nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],stddev=1.0 / math.sqrt(embedding_size)),name='Weights')
            tf.summary.histogram(name ='Weights', values = nce_weights)
        with tf.name_scope('Biases'):
            nce_biases = tf.Variable(tf.zeros([vocabulary_size]),dtype=tf.float32,name='Biases')
            tf.summary.histogram(name ='Biases', values = nce_biases)

    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss.
    with tf.name_scope('Loss'):
        output_layer = tf.nn.nce_loss(weights=nce_weights,
                                      biases=nce_biases,
                                      inputs=embed,
                                      labels=train_labels,
                                      num_sampled=num_sampled,
                                      num_classes=vocabulary_size)
        loss = tf.reduce_mean(output_layer)
        tf.summary.scalar('loss', loss)
    # Construct the SGD optimizer using a learning rate of 1.0.
    with tf.name_scope('Optimizer'):
        #optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.03).minimize(loss)


    # Compute the cosine similarity between minibatch examples and all embeddings.
    with tf.name_scope('normalized'):
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    '''
    with tf.name_scope('prediction'):
        similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)
        prediction = tf.nn.softmax(similarity)
    '''
    # Add variable initializer.
    merged = tf.summary.merge_all()
    init = tf.global_variables_initializer()


def writeLog(name,text):
    f = open("./NearestWord/"+name.encode('utf-8')+".txt","aw+")
    f.write(text.encode('utf-8'))
    f.write('\n')
    f.close()



def result_Json(final_embeddings,dictionary,idx=0,componentsNum =300,filename='images/tsne3.png'):

    logging.info("result_Json process start.")
    logging.debug(str(final_embeddings.shape))
    assert int(final_embeddings.shape[0])>int(componentsNum)
    idx = str(idx)
    outputText="./output/outputWord2Vec(v"+str(componentsNum)+")("+idx+").txt"
    #tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    #low_dim_embs = tsne.fit_transform(final_embeddings)
    pca = PCA(n_components=int(componentsNum))
    low_dim_embs = pca.fit_transform(final_embeddings)
    re_list={}
    lenDoct = len(dictionary)
    for i in range(lenDoct):
        #fetch = {dictionary[i]:low_dim_embs[i]}
        logging.debug('result_Json step:'+str(i)+'/'+str(lenDoct))
        #print(dictionary[i].encode('utf-8').decode('utf-8'))
        re_list[dictionary[i].encode('utf-8').decode('utf-8')] = low_dim_embs[i].tolist()
    f = open(outputText,'w')

    re_list = json.dumps(re_list, ensure_ascii=False)
    f.write(re_list)
    f.close()
    logging.info("result_Json process completed.")
logging.info("Begin training step.");

# Step 5-ex: Begin training.

'''
sess = tf.Session(graph=graph)
writer = tf.summary.FileWriter("TB/", graph = sess.graph)#TensorBoard
sess.run(init)
'''



num_steps = 100000
average_loss_num_step = 50

assert average_loss_num_step < num_steps
average_loss = 0
averagelossline_record=[]

with tf.Session(graph=graph) as sess:
    # We must initialize all variables before we use them.
    init.run()
    logging.info("Initialized")
    writer = tf.summary.FileWriter("TB/", graph = sess.graph)#TensorBoard

    for step in xrange(num_steps):
        batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
        showDetail("batch_inputs",batch_inputs)#Qusetion 128*1
        showDetail("batch_labels",batch_labels)#Answer 128*1

        nextDict = {train_inputs: batch_inputs, train_labels: batch_labels}

        stepMerged,stepLoss = sess.run([merged,loss],feed_dict=nextDict)
        logging.info("loss for this step("+str(step)+"):"+str(float(stepLoss)))
        writer.add_summary(stepMerged, step)
        average_loss += float(stepLoss)

        if step % average_loss_num_step == 0:
                if step > 0:
                    average_loss /= average_loss_num_step
                # The average loss is an estimate of the loss over the last 2000 batches.
                logging.info("Average loss at step "+str(step)+": "+str(float(average_loss)))
                average_loss = 0
        '''
        # Note that this is expensive (~20% slowdown if computed every 500 steps)
        if step % (average_loss_num_step*3) == 0:

            sim = similarity.eval(session=sess) # valid examples len * text dict len

            for i in xrange(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 50  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[:top_k]
                log_str = "Nearest to %s:" % valid_word
                for k in xrange(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = "%s %s," % (log_str, close_word)
                    writeLog(valid_word,close_word)
                print(log_str)
                #writeLog(valid_word,'--------')
        '''
        final_embeddings = normalized_embeddings.eval(session=sess)
        if step % 50000 == 0:
            result_Json(final_embeddings,reverse_dictionary,str(step))

print("training closed")
'''
# Step 6: Visualize the embeddings.
def plot_with_labels(low_dim_embs, labels, filename='images/tsne3.png',fonts=None):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"


    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                    fontproperties=fonts,
                    xy=(x, y),
                    xytext=(5, 2),
                    textcoords='offset points',
                    ha='right',
                    va='bottom')
    plt.savefig(filename,dpi=800)


try:
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    #为了在图片上能显示出中文
    font = FontProperties(fname=r"./simsun.ttc", size=14)

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    #perplexity:
    #   float, optional (default: 30)
    #   The perplexity is related to the number of nearest neighbors that is used in other manifold learning algorithms. Larger datasets usually require a larger perplexity. Consider selecting a value between 5 and 50. The choice is not extremely critical since t-SNE is quite insensitive to this parameter.

    #n_components :
    #   int, optional (default: 2)
    #   Dimension of the embedded space.

    #init:
    #   Initialization of embedding. Possible options are ‘random’, ‘pca’, and a numpy array of shape (n_samples, n_components). PCA initialization cannot be used with precomputed distances and is usually more globally stable than random initialization.

    #n_iter :
    #   int, optional (default: 1000)
    #   Maximum number of iterations for the optimization. Should be at least 250.

    plot_only = 500
    #showDetail("final_embeddings",final_embeddings)
    #showDetail("final_embeddings[0]",final_embeddings[0])
    final_embeddings_batch = final_embeddings[:plot_only, :]
    result_Json(final_embeddings,reverse_dictionary)

    low_dim_embs = tsne.fit_transform(final_embeddings_batch)
    #showDetail("low_dim_embs",low_dim_embs)

    labels = [reverse_dictionary[i] for i in xrange(plot_only)]

    #plot_with_labels(low_dim_embs, labels,fonts=font)



except ImportError:
    print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")



'''
endtime = datetime.datetime.now()
runtime=str((endtime-starttime).seconds)
print("runtime(s): ",runtime)
