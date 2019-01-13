# NLPLab

researching the core of nlp

----


## sentiment:
	implement of sentiment model in Chinese
----

## word2vec:
	implement of word2vec model

### word2vec/train_by_article.py
	1. loading the train data
		>  讀取停用字 loading stop words ( word2vec/stop_words.txt.py )
		>  loading training article ( word2vec/wiki/ or word2vec/TextForTrain/ )

		the main step:
		* clear special character:only chinese
		* simplified to treditional (../nstools/)

	2. Build the dictionary and replace rare words with UNKNOWWORD token.
		>  Build the dictionary
		>  rare words processed
		
		the main step:
		* Setting the size of the word set for the training model
		* using function: collections.Counter().most_common()
	3. Function to generate a training batch for the skip-gram model.
	4. Build and train a skip-gram model.
		> Loss: tf.nn.nce_loss()
		> Optimizer: tf.train.AdamOptimizer(learning_rate=1.0).minimize()
	5. Begin training
		> training stage
		> TensorBoard (will output to word2vec/TB/)
		> output to Json txt file :result_Json
----

## Crawler:
	Getting the date from website:
		1:scientific article
		2:Positive and negatiave review
## ModelTesting:
	the testing of model of nlp
----
## jieba_zn:
	Setting for traditional Chinese



## nstools:
	converting simplified to traditional

----
## TextRank:
	implement of TextRank model

## tf-idf-shortstr:
	implement of tf-idf model


	



----

### reference 
- https://codertw.com/%E7%A8%8B%E5%BC%8F%E8%AA%9E%E8%A8%80/575311/
- https://medium.com/pyladies-taiwan/%E8%87%AA%E7%84%B6%E8%AA%9E%E8%A8%80%E8%99%95%E7%90%86%E5%85%A5%E9%96%80-word2vec%E5%B0%8F%E5%AF%A6%E4%BD%9C-f8832d9677c8

