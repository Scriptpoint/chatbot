import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

#lammetize words to stem for shortenning
lemmatizer = WordNetLemmatizer()
#reading the json file
intents=json.loads(open('intents.json').read())
words=[]
tags=[]
docs=[]
ignore_letters=['?',',','.','!',':']
#finding user inputs from patterns in json file and slitting the words into individual words and storing it in the words list above
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list=nltk.word_tokenize(pattern)
        words.extend(word_list)
        #Upload the word list in document with respect to their tags i.e class and also append the tags in classes if it is not appended.this stores tags in classes and also word list in documents with their associated tags
        docs.append((word_list,intent['tag']))
        if intent['tag'] not in tags:
            tags.append(intent['tag'])
#lemmatize ignoring symbols
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
#avoiding duplication
words=sorted(set(words))
#avoid duplication of classes
tags=sorted(set(tags))
pickle.dump(words,open('words.pkl','wb'))
pickle.dump(tags,open('tags.pkl','wb'))

#neural network cannot work on words, so time to turn words into numerical values 0 and 1
training=[]
output_empty=[0] * len(tags)
for document in docs:
    #a bag for every document i.e tag and word combination
    bag=[]
    word_patterns=document[0]
    word_patterns=[lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

        output_row=list(output_empty)
        output_row[tags.index(document[1])]=1
        training.append([bag,output_row])
        #loop complete

#data processing
random.shuffle(training)
training=np.array(training)
train_x=list(training[:,0])
train_y=list(training[:,1])

#build neural network
model=Sequential()
#input layer of 130neurons and dense and activation with rectangular linear
model.add(Dense(130,input_shape=(len(train_x[0]),),activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
#SOFTMAX function sums up the output in output layer:how likely it is to have that output
model.add(Dense(len(train_y[0]),activation='softmax'))

#the lower part is the function to train the network
#SDG:Stochastic gradient descent is a gradient descent algorithm used for learning weights,parameters,coefficient of model like perceptron or linear regression.used to help neural network train on dataset
#lr=learning rate;default 0.1 or 0.01 ;but lower learning rate is superior , momentum=updates weight ; decay=reduces learning rate slowly until local minima is obtained
#nesterov accelerates decay if learning rate is large
sgd=SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])
#epoch=how many times to feed data to the neural network
variable=model.fit(np.array(train_x),np.array(train_y),epochs=200,batch_size=5,verbose=1)
model.save('Noramodel.h5',variable)
print("Done")
#Successful compilation is data trainede to the neural network

