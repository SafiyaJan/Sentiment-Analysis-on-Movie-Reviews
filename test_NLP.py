# import required packages
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.preprocessing import text, sequence
import matplotlib.pyplot as plt
import keras
import os
import re
import string


# YOUR IMPLEMENTATION
# Thoroughly comment your code to make it easy to follow


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

	# 1. Load your saved model
	model = keras.models.load_model("models/20868193_NLP_model")

	# 2. Load your testing data
	train, test = preprocess_data()

	train_labels, test_labels = create_labels()
	train, test, word_index = vectorize(train, test)

	# 3. Run prediction on the test data and print the test accuracy
	score = model.evaluate(test, test_labels, verbose=0)
	print("Test Accuracy:", score[1]*100)






