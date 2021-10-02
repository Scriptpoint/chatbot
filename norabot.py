import random
import numpy as np
import json
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

lemmatizer=WordNetLemmatizer()
intents=json.loads(open('intents.json').read())

words=pickle.load(open('words.pkl','rb'))
tags=pickle.load(open('tags.pkl','rb'))
model=load_model('Noramodel.h5')
def clean_sentence(sentence):
    sentence_words=nltk.word_tokenize(sentence)
    sentence_words=[lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words
#covert sentence to bag of words i.e like 0 and 1 numeric values in the training file
def bag_words(sentence):
    sentence_words=clean_sentence(sentence)
    bag=[0]*len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word==w:
                bag[i]=1
    return  np.array(bag)

def predict_tags(sentence):
    bagofwords=bag_words(sentence)
    result=model.predict(np.array([bagofwords]))[0]
    ERROR_THRESHOLD=0.25#25%
    results=[[i, r]for i,r in enumerate(result)if r> ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1],reverse=True)
    return_list=[]
    for r in results:
        return_list.append({'intent':tags[r[0]],'probability':str(r[1])})
    return return_list
def get_response(intents_list,intents_json):
    tag=intents_list[0]['intent']
    list_of_intents=intents_json['intent']
    for i in list_of_intents:
        if i['tag']==tag:
            result=random.choice(i['responses'])
            break
    return result
print("Nora is running")
while True:
    message=input("")
    ints=predict_tags(message)
    result=get_response(ints,intents)
    print(result)
