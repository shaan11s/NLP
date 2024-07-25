import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
stemmer = PorterStemmer()
lemmatizer= WordNetLemmatizer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def lem(word):
    return(lemmatizer.lemmatize(word.lower()))
    #return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, words):
    # lemmatize each word
    sentence_words = [lem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1
    return bag