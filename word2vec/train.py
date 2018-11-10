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

# -*- coding: utf-8 -*-
import re



# Step 1: Download the data.
# Read the data into a list of strings.
sourceText = '10day.txt'
def read_data():
    """
    对要训练的文本进行处理，最后把文本的内容的所有词放在一个列表中
    """
    #读取停用词
    stop_words = []
    with open('stop_words.txt',"r") as f:
        line = f.readline()
        while line:
            stop_words.append(line[:-1].decode('utf-8'))
            line = f.readline()
    stop_words = list(stop_words)
    print('停用詞讀取完畢，共{n}個詞'.format(n=len(stop_words)))

    # 读取文本，预处理，分词，得到词典
    raw_word_list = []
    with open(sourceText,"r") as f:#filter
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
words = read_data()
print('Data size', len(words))

# Step 2: Build the dictionary and replace rare words with UNKNOWWORD token.
use_value = 0.9 #max is 1
vocabulary_size = int(len(collections.Counter(words))*use_value) #55000
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
        if word in dictionary:
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
print('Most common words (+UNKNOWWORD)', count[:5])
''' print count and word
for i in count[1:20]:
    row=":"
    for i2 in i:
        try:
            row += i2.encode('utf-8')
        except:
            pass
    print(i2,row)
'''
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

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

def show_detail(T_name,T,detail=0):
    print('*'*50)
    print("name:",T_name)
    print("type:",type(T))
    try:
        print("len:",len(T))
    except:
        pass
    try:
        print("ndarray.shape:",T.shape)
    except:
        pass
    if(detail):
        print(T)





# Step 4: Build and train a skip-gram model.
batch_size = 128
embedding_size = 128
skip_window = 32   # 2*skip_window >= num_skips
num_skips = 2       # = batch_size/n
valid_size = 10
num_sampled = 64    # Number of negative examples to sample.

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import datetime
#plt.xlim(0,390)
#plt.ylim(0,300)

E_point = random.randint(valid_size,len(reverse_dictionary))


starttime = datetime.datetime.now()
#验证集
#valid_word = ['蕭炎','靈魂','火焰','蕭薰兒','藥老','天階',"雲嵐宗","烏坦城","驚詫"]
#E_point = random.randint(valid_size,len(reverse_dictionary))

valid_word=[]
topic_train=''
for i in xrange(E_point-valid_size,E_point):
    valid_word.append(reverse_dictionary[i].encode('utf-8'))
    topic_train+=reverse_dictionary[i].encode('utf-8')
    topic_train+=' | '

#show_detail("valid_word",valid_word)

valid_examples =[dictionary[li.decode('utf-8')] for li in valid_word]

graph = tf.Graph()
with graph.as_default():
    # Input data.
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # Ops and variables pinned to the CPU because of missing GPU implementation
    with tf.device('/cpu:0'):
        # Look up embeddings for inputs.
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        # Construct the variables for the NCE loss
        with tf.name_scope('Weights'):
            nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],stddev=1.0 / math.sqrt(embedding_size)))
        with tf.name_scope('Biases'):
            nce_biases = tf.Variable(tf.zeros([vocabulary_size]),dtype=tf.float32)

    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss.
    with tf.name_scope('loss'):
        output_layer = tf.nn.nce_loss(weights=nce_weights,biases=nce_biases,inputs=embed,labels=train_labels,num_sampled=num_sampled,num_classes=vocabulary_size)
        loss = tf.reduce_mean(output_layer)
        tf.summary.scalar('loss', loss)
    # Construct the SGD optimizer using a learning rate of 1.0.
    with tf.name_scope('Train'):
        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
        train_step = [optimizer, loss]
    # Compute the cosine similarity between minibatch examples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.nn.softmax(tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True))

    # Add variable initializer.
    init = tf.global_variables_initializer()



# Step 5-ex: Begin training.
#train_step = [optimizer, loss]
sess = tf.Session(graph=graph)
# 將視覺化輸出
#writer = tf.summary.FileWriter("TensorBoard/", graph = sess.graph)#TensorBoard
writer = tf.summary.FileWriter("TB/", graph = sess.graph)#TensorBoard
sess.run(init)

num_turn =10
num_steps = num_turn*100
average_loss = 0
average_loss_num_step = num_steps/10
averagelossline_record=[]
for step in xrange(num_steps):
    #sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
    #show_detail("batch_inputs",batch_inputs)#Qusetion 128*1
    #show_detail("batch_labels",batch_labels)#Answer 128*1

    feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
    _, loss_val = sess.run(train_step, feed_dict=feed_dict)
    average_loss += loss_val

    if step % average_loss_num_step == 0:
        if step > 0:
            average_loss /= average_loss_num_step
            # The average loss is an estimate of the loss over the last 2000 batches.
            print("Average loss at step ", step, ": ", average_loss)
            averagelossline_record.append(float(average_loss))
            average_loss = 0

        # Note that this is expensive (~20% slowdown if computed every 500 steps)
        if step % 10000 == 0:
            sim = similarity.eval(session=sess)
            for i in xrange(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[:top_k]
                log_str = "Nearest to %s:" % valid_word
                for k in xrange(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = "%s %s," % (log_str, close_word)
                print(log_str)
    final_embeddings = normalized_embeddings.eval(session=sess)


print("training closed")

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

outputText = "outputWord2Vec.txt"
def result_Json(final_embeddings,dictionary,filename='images/tsne3.png'):

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)

    low_dim_embs = tsne.fit_transform(final_embeddings)
    show_detail("low_dim_embs",low_dim_embs)
    show_detail("dictionary",dictionary)
    re_list={}
    for i in range(len(dictionary)):
        #fetch = {dictionary[i]:low_dim_embs[i]}
        re_list[dictionary[i].encode('utf-8')] = low_dim_embs[i].tolist()
    show_detail("re_list",re_list)
    f = open(outputText,'wa')
    import json
    re_list = json.dumps(re_list, ensure_ascii=False)
    f.write(re_list)
    f.close()
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
    #show_detail("final_embeddings",final_embeddings)
    #show_detail("final_embeddings[0]",final_embeddings[0])
    final_embeddings_batch = final_embeddings[:plot_only, :]
    #result_Json(final_embeddings,reverse_dictionary)

    low_dim_embs = tsne.fit_transform(final_embeddings_batch)
    #show_detail("low_dim_embs",low_dim_embs)

    labels = [reverse_dictionary[i] for i in xrange(plot_only)]

    #plot_with_labels(low_dim_embs, labels,fonts=font)



except ImportError:
    print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")


endtime = datetime.datetime.now()
runtime=str((endtime-starttime).seconds)
print("runtime(s): ",runtime)
