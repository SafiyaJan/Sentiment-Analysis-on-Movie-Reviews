import keras
import pickle
import string
import re
import os
import tarfile
import pandas as pd
import numpy as np
from keras.utils import get_file
from keras.preprocessing import text, sequence
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Embedding, LSTM, Dropout, Flatten
import matplotlib.pyplot as plt

# extract the tar file and save into directory
def extract_data():

  directory = get_file('aclImdb_v1.tar.gz', 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz', cache_subdir = "datasets",hash_algorithm = "auto", extract = True, archive_format = "auto")
 
  tar = tarfile.open(directory)
  tar.extractall('./data/') # specify which folder to extract to
  tar.close()

# read every review under path and save the content in an array
def read_file_into_array(path):

  review_array = []

  # for each file in folder
  for file in os.listdir(path):
    
    file_path = os.path.join(path,file)
    
    # open review
    f = open(file_path,'r')

    # read review and append to review array
    review_array.append(f.read().strip())
    
    # close file
    f.close()

  # return all reviews
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


  return train_pos + train_neg, test_pos + test_neg



# convert each review into a sequence of numbers 
def vectorize(train, test):

  # create vocab with 20000 words
  tokenizer = text.Tokenizer(20000)
  tokenizer.fit_on_texts(train)

  # tokenize all reviews in train and test set
  training = tokenizer.texts_to_sequences(train)
  testing = tokenizer.texts_to_sequences(test)

  # set the max length of reviews to be 1500
  max_len = 1500

  # convert review into a numerical representation
  training = sequence.pad_sequences(training,max_len)
  testing = sequence.pad_sequences(testing,max_len)

  return training, testing, tokenizer.word_index

# generate labels for eac
def create_labels():

  train_labels = []
  test_labels = []
  
  for i in range(25000):
    
    # positive reviews are labelled with 1
    if (i < 12500):
      train_labels.append(1.0)
      test_labels.append(1.0)

    # positive reviews are labelled with 0
    else:
      train_labels.append(0.0)
      test_labels.append(0.0)
      
  # return labels
  return np.asarray(train_labels, dtype=np.float), np.asarray(test_labels,dtype=np.float)

if __name__ == "__main__": 

	# 1. load your training data
  print ("Loading data and preprocessing data...")

  train, test = preprocess_data()
  train_labels, test_labels = create_labels()
  
  print ("Vectorizing data... ")

  train, test, word_index = vectorize(train, test)

  print ("Data pre-processing complete... ")

	# 2. Train your network
	# 		Make sure to print your training loss and accuracy within training to show progress
	# 		Make sure you print the final training accuracy

  print ("Building and Training model now... ")
  # Make model 

  model = Sequential()
  model.add(Embedding(20000,64,input_length=1500))
  model.add(Dropout(0.8))
  model.add(Flatten())
  model.add(Dense(1,activation='sigmoid'))

  model.compile(loss = 'binary_crossentropy', optimizer = Adam(learning_rate=0.0002), metrics = ['accuracy'])

  history = model.fit(train,train_labels,epochs=10, verbose=1, validation_data = (test,test_labels))

  print("Training Loss - :", history.history['loss'][-1])
  print("Training Accuracy - :", history.history['accuracy'][-1])

  
  # 3. Save the model
  model.save("models/20868193_NLP_model")



  # print("Testing Loss - :", history.history['val_loss'][-1])
  # print("Testing Accuracy - :", history.history['val_accuracy'][-1])

  # Plots for accuracy and loss

  # plt.plot(history.history['accuracy'])
  # plt.plot(history.history['val_accuracy'])
  # plt.title('Final NLP Model Testing and Training Accuracy v/s Epochs')
  # plt.ylabel('Accuracy')
  # plt.xlabel('Epoch')
  # plt.legend(['Training','Testing'])
  # plt.grid()
  # plt.show()

  # plt.plot(history.history['loss'])
  # plt.plot(history.history['val_loss'])
  # plt.title('Final NLP Model Testing and Training Loss v/s Epochs')
  # plt.ylabel('Loss')
  # plt.xlabel('Epoch')
  # plt.legend(['Training','Testing'])
  # plt.grid()
  # plt.show()
  






