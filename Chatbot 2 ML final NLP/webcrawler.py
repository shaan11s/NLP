from bs4 import BeautifulSoup
import requests
from urllib.parse import urlparse
import urllib.parse 
import urllib.request
import re
import nltk
from nltk.text import Text
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download("stopwords")
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')# Compute sentiment labels
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
# Functions

def Sort_Tuple(tup):
 
    # getting length of list of tuples
    lst = len(tup)
    for i in range(0, lst):
 
        for j in range(0, lst-i-1):
            if (tup[j][1] > tup[j + 1][1]):
                temp = tup[j]
                tup[j] = tup[j + 1]
                tup[j + 1] = temp
    return tup

def lexical_diversity(text): 
    return len(set(text)) / len(text) 

starter_url = "https://en.wikipedia.org/wiki/Neymar"

r = requests.get(starter_url)

data = r.text
soup = BeautifulSoup(data, 'html5lib')

# write urls to a file. 
# So we are basically taking a link, getting the links from there
# and checking if they fit our need
print("Getting links... \nSaving to urls.txt...")

with open('urls.txt', 'w') as f:
    # a tag is for links in html
    counter = 0
    netlocList = []
    for link in soup.find_all('a'):
        flag = 0
       # print(flag)
        
        #href is the actual link like we see in web browser
        link_str = str(link.get('href'))
        #print(link_str)

        temp = urlparse(link_str).netloc
        
        if not temp in netlocList:
            netlocList.append(temp)
            #print(netlocList)

            if 'Neymar' in link_str or 'neymar' in link_str:
                if link_str.startswith('/url?q='):
                    link_str = link_str[7:]
                    print('MOD:', link_str)
                if '&' in link_str:
                    i = link_str.find('&')
                    link_str = link_str[:i] # we keep link_string up to the &
                if link_str.startswith('https://www.') and 'google' not in link_str:
                    #make a list of the URL before .net and if it is in there, do not add to file

                    # Catch 403 and 404 Errors, and skip
                    try: 
                        x = urllib.request.urlopen(link_str) 
                        #print(x.read()) 
                    except Exception as e : 
                        #print(str(e)) 
                        flag = 1
                        #print(flag)
                        
                    if flag == 0:
                        f.write(link_str + '\n')
                        counter += 1

            if counter > 25:
                break

f.close()
# End of crawler
# print("end of crawler")

#Get the text from the URLs and clean (phase 1) data
print("Cleaning and saving data...")

fileNameList = ['data1.txt','data2.txt','data3.txt','data4.txt','data5.txt','data6.txt','data7.txt','data8.txt','data9.txt','data10.txt','data11.txt','data12.txt','data13.txt','data14.txt','data15.txt','data16.txt','data17.txt','data18.txt','data19.txt','data20.txt','data21.txt','data22.txt','data23.txt','data24.txt','data25.txt']
with(open('urls.txt', 'r') as f):
    with(open('data.txt', 'w') as fAll):
        for fileName in fileNameList:
            with(open(fileName, 'w') as fp):

                my_url = f.readline() #read a URL from file
                html = urllib.request.urlopen(my_url) 
                soup = BeautifulSoup(html, "html5lib") #get SOUP html
                data4 = soup.findAll('p') #get only <p> tagged elements
                data = str(data4) 
                #function to remove tags
                CLEANR = re.compile('<.*?>') 
                def cleanhtml(raw_html):
                    cleantext = re.sub('</p>.*?<p>', ' ',raw_html ) #anything between these should be deleted. </p>, <p>
                    cleantext = re.sub(CLEANR, '', cleantext)
                    cleantext = re.sub('\[', '', cleantext)
                    cleantext = re.sub('\]', '', cleantext)
                    return cleantext
                
                #tokenize and remove tags with above function
                temp = cleanhtml(data)
                tokens = []
                tokens = sent_tokenize(temp, language = "english") 
                
                #write to file
                for token in tokens:
                    token = str(token)
                    if ".com" in token:
                        continue
                    if "," == token:
                        continue
                    if ", " == token:
                        continue
                    if "    , " == token:
                        continue
                    if "" == token:
                        continue
                    if "            " in token:
                        token.replace(token[:12], '')
                    token = token.strip()
                    fp.write(token + "\n")
                    fAll.write(token + "\n")
                fp.close()

#print("Done!")
#all files written to
f.close()
fAll.close()

#TFIDF
# Get the corpus words set
print("TFID time...")
file = open("data.txt", "r")
f = file.read()
tokens = []
tokens = word_tokenize(f) 
allWords = tokens
words = [w.lower() for w in tokens]

# Remove all words in NLTK stopword list
stop_words = set(stopwords.words('english'))
for word in words:
    if word in stop_words:
        words.remove(word)

# Remove non alphabetical words
for word in words: 
    for char in word: 
        if not (char.isalpha()): 
           words.remove(word) 
           break
uniquewords = set(words)

dictsList = []
tokensList = []
for file in fileNameList:
    f = open(file)
    fsents = f.read()
    tokens = word_tokenize(fsents)
    words = [w.lower() for w in tokens]
    tokensList += words
    thisdict = dict.fromkeys(uniquewords, 0)
    for word in words:
        if word in thisdict:
            thisdict[word] +=1
    dictsList+=[thisdict]


def computeTF(wordDict, bagOfWords):
    tfDict = {}
    bagOfWordsCount = len(bagOfWords)
    for word, count in wordDict.items():
        tfDict[word] = count / float(bagOfWordsCount)
    return tfDict

listofTF = []
for i in range(len(dictsList)):
    tf = computeTF(dictsList[i],tokensList[i])
    listofTF += [tf]

def computeIDF(documents):
    import math
    N = len(documents)

    idfDict = dict.fromkeys(documents[0].keys(),0)
    for document in documents:
        for word, val in document.items():
            if val > 0:
                idfDict[word] += 1
    for word, val, in idfDict.items():
        idfDict[word] = math.log(N /float(val))
    return idfDict
    
idfs = computeIDF(dictsList) # this is where the argument should b every file

def computeTFIDF(tfBagOfWords, idfs):
    tfidf = {}
    for word, val in tfBagOfWords.items():
        tfidf[word] = val * idfs[word]
    return tfidf

tfidList = []
for i in range(len(listofTF)):
    tfidfA = computeTFIDF(listofTF[i], idfs) # all of them
    tfidList += [tfidfA]
    
    df = pd.DataFrame(tfidList)
    df.to_excel("tfid.xlsx")

#print(df)
pickle.dump(df, open('Df.pkl', 'wb'))


#CREATE DICT WITH TOP 15 KEYWORDS
#neymar, goals, psg, barcelona, brazil, santos, history, injury, contract, hilal
#messi, mbappe, suarez, libertadores, fans
thisdict = {
"neymar": [],
"goals": [],
"psg": [],
"barcelona": [],
"brazil": [],
"santos": [],
"history": [],
"injury": [],
"contract": [],
"hilal": [],
"messi": [],
"mbappe": [],
"suarez": [],
"libertadores": [],
"fans": []
}
file = open("data.txt", "r") 
f = file.read()

tokens = []
tokens = word_tokenize(f) 
words = [w.lower() for w in tokens]

datatemp = ""
for word in words:
    datatemp+= (word + " ")

sents = sent_tokenize(datatemp)

for sent in sents:
    if len(sent) > 5:
        for key in thisdict.keys():
            if key in sent:
                thisdict[key]+= [sent]

        

#print(thisdict)
pickle.dump(thisdict, open('Data.pkl', 'wb'))
file.close()

f = open("dataPICKLE.txt", "w") 
f.write(str(thisdict))
print("Done! Program finished.")