import csv
from nltk.tokenize import sent_tokenize, word_tokenize
import string
import torch
import json
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class ELMO_Model (nn.Module):
    def __init__(self,hidden_dim,vocab):
        super(ELMO_Model, self).__init__()
        pretrained_embeddings = torch.load('skip-gram-word-vectors.pt')
        self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, padding_idx=0, freeze=True)
        self.input_dim = 128            # dimension of pretrained skip-gram embeddings = 128
        self.hidden_dim = hidden_dim
        self.output_dim = len(vocab)

        self.bilstm_l1 = nn.LSTM(self.input_dim,self.hidden_dim,batch_first=True,bidirectional=True)
        self.bilstm_l2 = nn.LSTM(self.hidden_dim*2, self.hidden_dim, batch_first=True,bidirectional=True)

        self.final_layer = nn.Linear(self.hidden_dim*2, self.output_dim)

    def forward(self,x):
        pretrained_embds = self.embedding(x)
        bilstm_l1, _ = self.bilstm_l1(pretrained_embds)
        bilstm_l2, _ = self.bilstm_l2(bilstm_l1)
        elmo_embds = (1/3*pretrained_embds) + (1/3*bilstm_l1) + (1/3*bilstm_l2)
        word_embds = self.final_layer(elmo_embds)
        return word_embds

###################################################################################

def generate_training_data (vocab,tokenized_data):
    train_data = []
    for sent_tokens in tokenized_data:
        word_data = []
        for word_token in sent_tokens:
            word_data.append(vocab[word_token])
            if len(word_data) >= 50:
                break
        word_data = word_data + [0]*(50-len(word_data))
        train_data.append(word_data)
        if len(train_data) == 15000:
            break
    training_data = torch.tensor(train_data)
    return training_data

###################################################################################

def train_model(model, data_loader, vocab_size, batch_size=100, num_epochs=5, learning_rate=0.01):
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        i = 0
        for batch in data_loader:
            print(i)
            i += 1
            optimizer.zero_grad()
            embds=model(batch)
            embds=embds.view(-1, embds.size(-1))
            batch=batch.view(-1)
            loss=criterion(embds, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss / len(data_loader)}")

    return model


###################################################################################

def main ():
    with open ('dataset/train.csv', 'r') as file:
        dataset = []
        file_reader = csv.reader(file)
        i = 0
        for row in file_reader:
            if i > 0:
                dataset.append(row[1])
            i += 1
            if i > 15000:
                break

    tokenized_data = []
    for desc in dataset:
        sentences = sent_tokenize(desc)
        for sentence in sentences:
            tokens = word_tokenize(sentence.lower())
            tokens_no_punct = [token for token in tokens if token not in string.punctuation]
            tokenized_data.append(tokens_no_punct)

    # vocab = {}             # word with corresponding index
    # word_freq = {}         # words with corresponding frequencies
    # idx = 0
    # for i in range (0,len(tokenized_data)):
    #     tokenized_data[i] = ['<s>'] + tokenized_data[i] + ['</s>']
    #     for word in tokenized_data[i]:
    #         if word not in vocab:
    #             word_freq[word] = 1
    #             vocab[word] = idx
    #             idx += 1
    #         else :
    #             word_freq[word] += 1

    # vocab_dict = {'vocab': vocab, 'word_freq': word_freq}
    # with open ('vocab_dict.json', 'w') as file:             # saving the vocab in a json file
    #     json.dump(vocab_dict,file)
    
    with open('vocab_dict.json', 'r') as file:
        data = json.load(file)
        vocab = data['vocab']
        word_freq = data['word_freq']

    training_data = generate_training_data(vocab,tokenized_data)
    data_loader = DataLoader(training_data, batch_size=100)

    # TRAINING THE ELMO MODEL
    # model = ELMO_Model(64, vocab)
    # model = train_model(model, data_loader, len(vocab))
    # torch.save(model, 'bilstm.pt')

    # LOADING ELMO MODEL 
    model = torch.load('bilstm.pt')

if __name__ == "__main__":
    main()
