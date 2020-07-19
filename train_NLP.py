from keras.utils import get_file
import tarfile
import os
import pandas as pd
import numpy as np
from keras.preprocessing import text, sequence
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from keras.layers import Dense, Embedding, LSTM, Dropout, Flatten
import string
import re
import matplotlib.pyplot as plt
import math
import keras

# extract the tar file and save into directory
def extract_data():
  print ("Hello")
  directory = get_file('aclImdb_v1.tar.gz', 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz', cache_subdir = "datasets",hash_algorithm = "auto", extract = True, archive_format = "auto")
  tar = tarfile.open(directory)
  tar.extractall('./data/') # specify which folder to extract to
  tar.close()

# read every review under path and save the content in an array
def read_file_into_array(path):
	review_array = []

	for file in os.listdir(path):
		file_path = os.path.join(path,file)
		f = open(file_path,'r')
		review_array.append(f.read().strip())
		f.close()

	return review_array

def preprocess_data():

  # get training + testing data
  train_pos = read_file_into_array("data/aclImdb/train/pos")
  train_neg = read_file_into_array("data/aclImdb/train/neg")
  test_pos = read_file_into_array("data/aclImdb/test/pos")
  test_neg = read_file_into_array("data/aclImdb/test/neg")
  
  for i in range(12500):

    # remove all html tags
    tag_rgx = re.compile(r'<[^>]+>')

    train_pos[i] = tag_rgx.sub('',train_pos[i])
    train_neg[i] = tag_rgx.sub('',train_neg[i])
    test_pos[i] = tag_rgx.sub('',test_pos[i])
    test_neg[i] = tag_rgx.sub('',test_neg[i])

    punctuation = '!\"#$%&\'()*+,-./:;<=>?@[\]^_`´{|}~'

    # remove all punctuations and make lowercase 
    train_pos[i] = train_pos[i].translate(str.maketrans('', '', punctuation)).lower()
    train_neg[i] = train_neg[i].translate(str.maketrans('', '', punctuation)).lower()
    test_pos[i] = test_pos[i].translate(str.maketrans('', '', punctuation)).lower()
    test_neg[i] = test_neg[i].translate(str.maketrans('', '', punctuation)).lower()

    # remove all numbers 
    train_pos[i] = train_pos[i].translate(str.maketrans('', '', string.digits))
    train_neg[i] = train_neg[i].translate(str.maketrans('', '', string.digits))
    test_pos[i] = test_pos[i].translate(str.maketrans('', '', string.digits))
    test_neg[i] = test_neg[i].translate(str.maketrans('', '', string.digits))


    # remove all stop words
    # text_tokens = word_tokenize(train_pos[i])
    # train_pos_tokens = [word for word in text_tokens if not word in stopwords.words()]
    # train_pos[i] = (" ").join(train_pos_tokens)

    # text_tokens = word_tokenize(train_neg[i])
    # train_neg_tokens = [word for word in text_tokens if not word in stopwords.words()]
    # train_neg[i] = (" ").join(train_neg_tokens)

    # text_tokens = word_tokenize(test_pos[i])
    # test_pos_tokens = [word for word in text_tokens if not word in stopwords.words()]
    # test_pos[i] = (" ").join(test_pos_tokens)

    # text_tokens = word_tokenize(test_neg[i])
    # test_neg_tokens = [word for word in text_tokens if not word in stopwords.words()]
    # test_neg[i] = (" ").join(test_neg_tokens)

  return train_pos + train_neg, test_pos + test_neg



# convert each review into a sequence of numbers 
def vectorize(train, test):

	tokenizer = text.Tokenizer(20000)
	tokenizer.fit_on_texts(train)

	training = tokenizer.texts_to_sequences(train)
	testing = tokenizer.texts_to_sequences(test)

	max_len = 1500

	training = sequence.pad_sequences(training,max_len)
	testing = sequence.pad_sequences(testing,max_len)

	return training, testing, tokenizer.word_index

# generate labels for eac
def create_labels():
  train_labels = []
  test_labels = []
  for i in range(25000):
    if (i < 12500):
      train_labels.append(1.0)
      test_labels.append(1.0)
    else:
      train_labels.append(0.0)
      test_labels.append(0.0)
      
  return np.asarray(train_labels, dtype=np.float), np.asarray(test_labels,dtype=np.float)

if __name__ == "__main__": 

	# 1. load your training data
  print ("Loading data")
  # extract_data()
  train, test = preprocess_data()

  train_labels, test_labels = create_labels()
  train, test, word_index = vectorize(train, test)


  print ("Done Loading data")

	# # 2. Train your network
	# # 		Make sure to print your training loss and accuracy within training to show progress
	# # 		Make sure you print the final training accuracy

  # BESTTTTTT
  # model = Sequential()
  # model.add(Embedding(20000,32,input_length=1500))
  # model.add(Dropout(0.8))
  # model.add(Flatten())
  # model.add(Dense(1,activation='sigmoid'))

  model = Sequential()
  model.add(Embedding(20000,32,input_length=1500))
  model.add(Dropout(0.8))
  model.add(Flatten())
  model.add(Dense(1,activation='sigmoid'))

  model.compile(loss = 'binary_crossentropy', optimizer = Adam(learning_rate=0.0002), metrics = ['accuracy'])

  history = model.fit(train,train_labels,epochs=15, verbose=1, validation_data = (test,test_labels))


  # save the model
  model.save("models/20868193_NLP_model")
  model = keras.models.load_model("20868193_NLP_model")
  score = model.evaluate(train, train_labels, verbose=0)
  print("Train Accuracy:", score[1])

  # score = model.evaluate(test, test_labels, verbose=0)
  # print("Test Accuracy:", score[1])







