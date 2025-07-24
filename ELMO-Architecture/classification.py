import torch
import torch.nn as nn
import csv
from nltk.tokenize import sent_tokenize, word_tokenize
import string
import json
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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

################################################################################

class LSTM_Model(nn.Module):
    def __init__(self,ELMO_model,embd_dim,hidden_dim,output_dim,batch_size):
        super(LSTM_Model,self).__init__()
        self.ELMO_model = ELMO_model
        self.embd_dim = embd_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.ELMO_model.requires_grad_(False)
        self.lstm_layer = nn.LSTM(self.embd_dim,self.hidden_dim,batch_first=True)
        self.linear_layer = nn.Linear(self.hidden_dim,self.output_dim)

        self.hidden = (torch.zeros(1,batch_size, self.hidden_dim),
                torch.zeros(1,batch_size, self.hidden_dim)) 
        
        self.lambda_params = nn.Parameter(torch.FloatTensor([1/3, 1/3, 1/3]))                              # Training to find best lambda's (4.1)
        # self.lambda_params = nn.Parameter(torch.FloatTensor([1/3, 1/3, 1/3]), requires_grad=False)       # Freezing Lambda Values (4.2)

        # self.lambda_params = nn.Sequential(
        #     nn.Linear((128+128+128), 4*64),                                                              # For Learning Function (4.3)
        #     nn.ReLU(),
        #     nn.Linear(4*64, 64*2),
        #     nn.ReLU()
        # )
        
    def forward(self, x):
        initial_layer = self.ELMO_model.embedding(x)  
        bilstm_l1, _ = self.ELMO_model.bilstm_l1(initial_layer)
        bilstm_l2, _ = self.ELMO_model.bilstm_l2(bilstm_l1)

        # concat_embds = torch.cat([initial_layer, bilstm_l1, bilstm_l2], dim=2)                           # For Learning Function (4.3)
        # word_embds = self.lambda_params(concat_embds)

        word_embds = self.lambda_params[0]*initial_layer + self.lambda_params[1]*bilstm_l1 + self.lambda_params[2]*bilstm_l2
        word_embds = word_embds.float()
        lstm_layer = self.lstm_layer(word_embds)
        hidden_l = lstm_layer[0][:,-1,:]
        final_layer = self.linear_layer(hidden_l)
        return final_layer,self.lambda_params

################################################################################
    
def generate_data (vocab,tokenized_data):
    train_data = []
    for sent_tokens in tokenized_data:
        word_data = []
        for word_token in sent_tokens:
            if word_token in vocab:
                word_data.append(vocab[word_token])
            else:
                word_data.append(vocab['<unk>'])
            if len(word_data) >= 50:
                break
        word_data = word_data + [0]*(50-len(word_data))
        train_data.append(word_data)
        if len(train_data) == 15000:
            break
    training_data = torch.tensor(train_data)
    return training_data

################################################################################

def prepare_data(file,vocab):
    with open (file, 'r') as data_file:
        cat = []
        dataset = []
        file_reader = csv.reader(data_file)
        i = 0
        for row in file_reader:
            if i > 0:
                cat.append(row[0])
                dataset.append(row[1])
            i += 1
            if i > 15000:
                break

    tokenized_data = []
    label_vector = []
    iter = 0
    for desc in dataset:
        arr = [0,0,0,0]
        arr[int(cat[iter])-1] = 1
        sentences = sent_tokenize(desc)
        for sentence in sentences:
            tokens = word_tokenize(sentence.lower())
            tokens_no_punct = [token for token in tokens if token not in string.punctuation]
            tokenized_data.append(tokens_no_punct)
            label_vector.append(arr)
        iter += 1

    X_desc = generate_data(vocab,tokenized_data)
    label_vector = label_vector[:15000]
    Y_labels = torch.tensor(label_vector, dtype=torch.float32)

    return X_desc,Y_labels

################################################################################

def train_classification_model (model,X_train,Y_train,batch_size,num_epochs,learning_rate):
    data = TensorDataset(X_train,Y_train)
    data_loader = DataLoader(data,batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        for x_desc, y_labels in data_loader:
            optimizer.zero_grad()
            y_pred,params = model(x_desc)
            loss = criterion(y_pred,y_labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    return model

################################################################################

def evaluate_model(model,x_desc, y_labels,batch_size):
    correct = 0
    total = 0
    predicted_labels = []
    true_labels = []
    data = TensorDataset(x_desc,y_labels)
    data_loader = DataLoader(data,batch_size)

    with torch.no_grad():
        for x_desc, y_labels in data_loader:
            y_pred,params = model(x_desc)
            _, predicted = torch.max(y_pred.data, 1)
            predicted_tuple = tuple(predicted.tolist())
            predicted_labels.extend(predicted_tuple)
            _, true = torch.max(y_labels.data, 1)
            true_tuples = tuple(true.tolist())
            true_labels.extend(true_tuples)
            for i in range(0, len(predicted_tuple)):
                if predicted_tuple[i] == true_tuples[i]:
                    correct += 1
                total += 1
            
    accuracy = correct / total
    print(f'Accuracy on the test set: {accuracy:.2%}')

    # Calculate precision, recall, and F1 score
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    print(f'Precision: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1 Score: {f1:.2f}')

    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    print('Confusion Matrix:')
    plt.figure(figsize=(15, 10))

    classes = [1,2,3,4]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, annot_kws={'size': 12})
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

################################################################################

def main ():
    elmo_model = torch.load('bilstm.pt')

    with open('vocab_dict.json', 'r') as file:
        data = json.load(file)
        vocab = data['vocab']

    X_train,Y_train = prepare_data('dataset/train.csv',vocab)
    X_test,Y_test = prepare_data('dataset/test.csv',vocab)

    embd_dim = 128
    hidden_dim = 256
    output_dim = 4
    batch_size = 50
    num_epochs = 50
    learning_rate = 0.005

    classification_model = LSTM_Model(elmo_model,embd_dim,hidden_dim,output_dim,batch_size)
    # classification_model = train_classification_model(classification_model,X_train,Y_train,batch_size,num_epochs,learning_rate)
    # torch.save(classification_model.state_dict(),'classifier.pt')
    classification_model.load_state_dict(torch.load('classifier.pt'))
    evaluate_model(classification_model,X_train,Y_train,batch_size)

if __name__ == "__main__":
    main()