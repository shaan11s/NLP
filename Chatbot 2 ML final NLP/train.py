import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
from nltk_utils import tokenize, lem, bag_of_words 
from nltk.corpus import wordnet as wn
import nltk
nltk.download('wordnet')

def get_synonyms(word):
    synonyms = set()
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())  # Add the synonyms found
    return list(synonyms)

def expand_with_synonyms(patterns):
    expanded_patterns = []
    for sentence in patterns:
        words = sentence.split()  # Split the sentence into words
        sentence_variants = [sentence]  # Include the original sentence
        
        for i, word in enumerate(words):
            synonyms = get_synonyms(word)
            for synonym in synonyms:
                if synonym != word:  # Ensure the synonym is different from the original word
                    new_sentence = ' '.join(words[:i] + [synonym] + words[i+1:])
                    sentence_variants.append(new_sentence)
        
        expanded_patterns.extend(sentence_variants)
    return list(set(expanded_patterns))  # Remove duplicates and return


 
# Define the RNN model class
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


# Load intents from JSON file
with open('intents.json', 'r') as f:
    intents = json.load(f)

# Data preparation
all_words = []
tags = []
xy = []
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    expanded_patterns = expand_with_synonyms(intent['patterns'])
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

ignore_words = ['?', '.', '!']
all_words = [lem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)
X_train = np.expand_dims(X_train, 1)  # Add a sequence dimension

# Splitting the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Define the dataset class
class ChatDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.n_samples = len(x_data)
        self.x_data = x_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

# Creating datasets and dataloaders
train_dataset = ChatDataset(X_train, y_train)
test_dataset = ChatDataset(X_test, y_test)
train_loader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=8, shuffle=False)

# Model, loss, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RNN(len(X_train[0][0]), 8, len(tags), num_layers=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Evaluation function to plot confusion matrix
def evaluate_model(loader):
    model.eval()
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for words, labels in loader:
            words = words.to(device)
            labels = labels.to(device)
            outputs = model(words)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=tags, yticklabels=tags)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    
# def evaluate_model(loader):
#     model.eval()
#     total_loss, total_correct, total = 0, 0, 0
#     for words, labels in loader:
#         words = words.to(device)
#         labels = labels.to(device)
#         with torch.no_grad():
#             outputs = model(words)
#             loss = criterion(outputs, labels)
#             _, predicted = torch.max(outputs, 1)
#             total_loss += loss.item()
#             total_correct += (predicted == labels).sum().item()
#             total += labels.size(0)
#     print(f'Validation Loss: {total_loss/len(loader):.4f}, Accuracy: {total_correct/total:.4f}')


# Training the model
num_epochs = 1000

from torch.optim.lr_scheduler import StepLR
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=50, gamma=0.1)  # Adjust learning rate every 50 epochs
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        outputs = model(words)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate on both training and test sets
print("Evaluating on training set:")
evaluate_model(train_loader)
print("Evaluating on test set:")
evaluate_model(test_loader)
print('Training complete.')

# Save the trained model and additional data necessary for the chat application
model_info = {
    'model_state': model.state_dict(),
    'input_size': len(all_words),
    'hidden_size': 8,  # This should match the hidden_size used when initializing your model
    'output_size': len(tags),
    'num_layers': 2,  # This should match the num_layers used in your RNN model
    'all_words': all_words,
    'tags': tags
}
torch.save(model_info, "data.pth")
