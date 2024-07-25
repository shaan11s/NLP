import random
import json
import torch
from nltk_utils import bag_of_words, tokenize
import torch.nn as nn
from nltk.stem import WordNetLemmatizer
import csv
import os
from datetime import datetime, timedelta, date
import pandas as pd
from googlesearch import search
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

def check_sentiment_vader(text):
    # Create a SentimentIntensityAnalyzer object
    sia = SentimentIntensityAnalyzer()
    
    # Get the sentiment scores
    sentiment_scores = sia.polarity_scores(text)
    #print(sentiment_scores)
    
    # Determine the sentiment from the compound score
    compound_score = sentiment_scores['compound']
    if compound_score >= 0.05:
        return "Positive"
    elif compound_score <= -0.05:
        return "Negative"
    else:
        return "Positive"

def find_user_by_name(filename, name_to_find):
    try:
        with open(filename, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)  # Read the header to get the column indices
            if 'Name' in header and 'Date' in header:
                name_index = header.index('Name')
                date_index = header.index('Date')
                
                # Search for the name in the 'Name' column
                for row in reader:
                    if row[name_index] == name_to_find:
                        return row[name_index], row[date_index]
                return None  # Return None if no match is found
            else:
                print("CSV file must include 'Name' and 'Date' columns")
    except FileNotFoundError:
        print(f"No such file: {filename}")
    except Exception as e:
        print(f"Empty File: {e}")

def write_user_data(filename, row, fields=['Name', 'Date', 'Soccer_Knowledge', 'Satisfaction']):
    file_exists = os.path.isfile(filename)
    
    # Open the file in append mode ('a') so data is added at the end of the file.
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # If the file doesn't exist or is empty, write the header.
        if not file_exists or os.stat(filename).st_size == 0:
            fields2=['Name', 'Date', 'Soccer_Knowledge', 'Satisfaction']
            writer.writerow(fields2)
        
        # Write the user data row.
        writer.writerow(row)

# Neural network class definition
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        out = self.fc(out[:, -1, :])  # Decode the hidden state of the last time step
        return out

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load intents
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

## Load model data
FILE = "data.pth"
data = torch.load(FILE)
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
num_layers = data.get("num_layers", 1)
model_state = data["model_state"]

# Initialize the RNN model
model = RNN(input_size, hidden_size, output_size, num_layers).to(device)

# Load the model state
model.load_state_dict(model_state)
model.eval()

all_words = data['all_words']
tags = data['tags']

# Lemmatizer setup
lemmatizer = WordNetLemmatizer()

# Lemmatization function
def lemmatize_words(words):
    return [lemmatizer.lemmatize(word) for word in words]

# Modified tokenize function that includes lemmatization
def tokenize_and_lemmatize(sentence):
    words = tokenize(sentence)
    return lemmatize_words(words)

#Pre Model Bot Interaction 
user_row = ['null', 'null', 'null', 'null']

print("Santos: Hi! I am Santos, a Neymar fan bot! I love to talk about Neymar.")
print("Santos: If you want to quit, just type quit and press enter!")
print("Santos: What is your name?")
today = date.today()
user_row[1] = today
name = input(": ")
user_row[0] = name
if name =="quit":
    print("BYE!")
    exit()
# If user name is in user_data, bring the user up and say his name and date
# Example usage
result = find_user_by_name('user_data.csv', name)
if result:
    print("User found:", result)
    delay = timedelta(seconds=2)
    endtime = datetime.now() + delay
    while datetime.now() < endtime:
        pass
    greetings = ["Welcome back,", "We missed you,", "It's been a while,"]
    greeting = random.choice(greetings)
    print("Santos:",greeting, name+"!")
else:
    print("New User!")
    delay = timedelta(seconds=2)
    endtime = datetime.now() + delay
    while datetime.now() < endtime:
        pass
    greetings = ["Hi,", "Hey there,", "Whats up,"]
    greeting = random.choice(greetings)
    print("Santos:",greeting, name+"!")

delay = timedelta(seconds=2)
endtime = datetime.now() + delay
while datetime.now() < endtime:
    pass

# LIVE LOOKUP
print("Would you like me to pull a Live Table and check to see how Neymars team is doing now?")
input_table = input(": ")
if input_table =="quit":
        print("BYE!")
        exit()
x1 = check_sentiment_vader(input_table)
#if input sentiment is good, show table, else do not!
if x1 == "Positive":
    print("Here you go! \n")
    url = 'https://www.espn.com/soccer/table/_/league/ksa.1'
    all_tables = pd.read_html(url)
    all_tables[0].to_csv('ksaTable.csv', index=False) #record live look up of table
    print(all_tables[0])
    print("Do you want to see the stats?")
    response = input(": ")
    if response =="quit":
        print("BYE!")
        exit()
    x2 = check_sentiment_vader(input_table)
    if x2 == "Positive":
        print(all_tables[1])

    with open('ksaTable.csv') as csvfile:
        csv_reader = csv.reader(csvfile)
        for index, row in enumerate(csv_reader):
            if index == 1:
                if row[0] == '1HILAl Hilal':
                    print("\nSantos: Looks like Neymars team is currently first place!")
                    print("Santos: We would expecet nothing less from the best!\n")
                if row[0] == '2HILAl Hilal':
                    print("Santos: Looks like Neymars team is currently second place!")
                    print("Santos: Not bad..\n")
                if row[0] == '3HILAl Hilal':
                    print("Santos: Looks like Neymars team is currently third place!")
                    print("Santos: Not bad..\n")
else:
    print("No problem!\n")

# Bot interaction
bot_name = "Santos"
flag = 0
soccer_knowledge = 0
user_row[2] = soccer_knowledge
print("Let's chat! (type 'quit' to exit)")
print("You can ask me anything about Neymar!")
while True:
    sentence = input(": ")
    sentence_save = sentence
    if sentence == "quit":
        write_user_data('user_data.csv', user_row)
        break

    sentence = tokenize_and_lemmatize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])  # Reshape to make it 2-D (1, input_size)
    X = torch.from_numpy(X).to(device)
    X = X.unsqueeze(0)  # Add a batch dimension, now X becomes 3-D (1, 1, input_size)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    print(prob.item())  # DEBUG
    prob_save = prob.item()
    if prob.item() > 0.45:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                if tag in ["play_style","Barcelona","PSG","Santos"]:
                    soccer_knowledge = 1
                    user_row[2] = soccer_knowledge

                print(f"{bot_name}: {random.choice(intent['responses'])}")
                answer_save = f"{bot_name}: {random.choice(intent['responses'])}"
    else:
        print(f"{bot_name}: I do not understand... Here are some resources to help!")
        query = sentence_save
        for j in search(query, tld="co.in", num=10, stop=10, pause=2):
            print(j)

    if random.random() < 0.4:
        if flag == 0: 
            flag = 1
            print("\n\n")
            print("How satisfied are you with your answer? (1-5: with 5 being perfect)")
            print("\n\n")
            response = input(": ")
            response_save = response
            temp_str = "    Response: " + str(response_save) + "    Probability: " + str(prob_save) + "    Question: " + str(sentence_save) + "    Answer: " + str(answer_save)
            user_row[3] = temp_str

