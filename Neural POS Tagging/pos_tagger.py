import torch
from torch import nn,optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support,confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import re
import argparse
import numpy as np

def parse_file (path):
# Returns a list which containings tokenized sentences with respective POS tags for the words
# Format - [[[words of sentence i],[POS tags for the words in sentence i]],...]
    data = []
    with open (path, 'r', encoding='utf-8') as file:
        words = []
        tags = []
        for line in file:
            line = line.strip()                   # Removing the leading and trailing Blankspaces.
            if line and line[0].isdigit():
                parts = line.split("\t")
                if parts[0] == '1' and len(words) != 0:
                    sentence_data = [words, tags]
                    data.append(sentence_data)
                    words = [parts[1]]
                    tags = [parts[3]]
                else :
                    words.append(parts[1])
                    tags.append(parts[3])
    return data

dev_data = parse_file("UD_English-Atis/en_atis-ud-dev.conllu")
train_data = parse_file("UD_English-Atis/en_atis-ud-train.conllu")
test_data = parse_file("UD_English-Atis/en_atis-ud-test.conllu")

# Creating a word index and tag index dictionary
def get_index_dict (data):
    k = 0
    j = 0
    vocab = {}
    tags = {}
    for sentence in data:
        for i in range (0,len(sentence[0])):
            if sentence[0][i] not in vocab:
                vocab[sentence[0][i]] = k
                k += 1
            if sentence[1][i] not in tags:
                tags[sentence[1][i]] = j
                j += 1
    vocab['<unk>'] = k
    vocab['<s>'] = k+1
    vocab['<\s>'] = k+2
    vocab['<pad>'] = k+3

    tags['START'] = j
    tags['END'] = j+1
    tags['<pad>'] = j+2

    idx_tags = {}
    for tag in tags:
        idx_tags[tags[tag]] = tag

    return vocab,tags,idx_tags

vocab,tags,idx_tags = get_index_dict(train_data)

# Adding few unknow values to training data 
def handling_unknowns (train):
    flag = 0
    for sent_d in train:
        if flag % 25 == 0:
            rand_int = np.random.randint(0, len(sent_d[0]))
            train[flag][0][rand_int] = 'unk'                       # 'unk' is not a part of vocab
        flag += 1

    return train

train_data = handling_unknowns(train_data)

#############################################################################################################################################################

def remove_punctuation(sentence):
    punctuation_pattern = r'[^\w\s]'
    sentence_without_punctuation = re.sub(punctuation_pattern, '', sentence)
    return sentence_without_punctuation

#############################################################################################################################################################

# Preparing the Training Data
def preparing_data (vocab,tags,data,p,s):
    X = []
    Y = []
    start = ['<s>']*p
    end = ['<\s>']*s
    start_tags = ['START']*p
    end_tags = ['END']*s
    for sentence_data in data:
        sentence_data[0] = start + sentence_data[0] + end
        sentence_data[1] = start_tags + sentence_data[1] + end_tags

        word2idx = []
        tag2idx = []
        for i in range (0,len(sentence_data[0])):
            if sentence_data[1][i] in tags:
                tag2idx.append(tags[sentence_data[1][i]])
                if sentence_data[0][i] in vocab:
                    word2idx.append(vocab[sentence_data[0][i]])
                else :
                    word2idx.append(vocab['<unk>'])
        
        for i in range(0,len(word2idx)-s-p):
            word_window = word2idx[i:(p+s+1+i)]
            X.append(word_window)
            y_vect = []
            for j in range(0,p+s+1):
                y = [float(0)]*len(tags)
                y[tag2idx[i+j]] = float(1)
                y_vect.append(y)
            Y.append(y_vect)

    ipt = torch.tensor(X)
    opt = torch.tensor(Y)

    return ipt,opt

# p = 4 and s = 4
p = 4
s = 4
X_train,Y_train = preparing_data(vocab,tags,train_data,p,s)
X_test, Y_test = preparing_data(vocab,tags,dev_data,p,s)

#############################################################################################################################################################

def Accuracy (y_pred,y_actual):
    correct = 0
    for i in range (0,y_pred.size(0)):
        if torch.equal(y_pred[i][4], y_actual[i][4]):
            correct += 1

    return (correct/y_pred.size(0))

##############################################################################################################################################################

def Model_Evaluate (model,ipt,opt):
    model.eval()
    with torch.inference_mode():
        pred_opt = torch.round(torch.sigmoid(model(ipt).squeeze()))

    print(f"Accuracy : {Accuracy(pred_opt,opt)}")
    pred_opt_np = pred_opt[:,4,:].detach().numpy()
    opt_np = opt[:,4,:].detach().numpy()

    _, recall_micro, f1_score_micro, _ = precision_recall_fscore_support(opt_np, pred_opt_np, average='micro',zero_division=1)
    _, recall_macro, f1_score_macro, _ = precision_recall_fscore_support(opt_np, pred_opt_np, average='macro',zero_division=1)

    print("Micro-average Recall:", recall_micro)
    print("Micro-average F1-score:", f1_score_micro)
    print("Macro-average Recall:", recall_macro)
    print("Macro-average F1-score:", f1_score_macro)

    opt_labels = np.argmax(opt_np, axis=1)
    pred_labels = np.argmax(pred_opt_np, axis=1)
    
    confusion_mat = confusion_matrix(opt_labels,pred_labels)
    print("Confusion Matrix:")
    plt.figure(figsize=(15, 10))

    pos_vocab = []
    for tag in tags:
        pos_vocab.append(tag)

    # Displaying the confusion matrix as a heatmap
    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', xticklabels=pos_vocab, yticklabels=pos_vocab, annot_kws={'size': 12})
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

##############################################################################################################################################################
    
def tag_sentence (model,sentence):
    input_sent = remove_punctuation(sentence).lower().split()
    sent = p*['<s>'] + input_sent + s*['</s>']
    index = []
    for word in sent:
        if word in vocab:
            index.append(vocab[word])
        else:
            index.append(vocab['<unk>'])
    ipt = []
    for i in range(0,len(index)-s-p):
        word_window = index[i:(p+s+1+i)]
        ipt.append(word_window)
    
    ipt_sent = torch.tensor(ipt)
    model.eval()
    with torch.inference_mode():
        pred_opt = torch.round(torch.sigmoid(model(ipt_sent).squeeze()))

    opt = pred_opt[:,p,:]
    word_no = 0
    for tag in opt:
        flag = 0
        for i in range(0,len(tag)):
            if int(tag[i]) == 1:
                flag = 1
                print(f"{input_sent[word_no]} {idx_tags[i]}")
                break
        if flag == 0:
            print(f"{input_sent[word_no]} NOUN")
        word_no += 1

##############################################################################################################################################################

class FFNN_POS_Tagger (nn.Module):
    def __init__ (self,vocab_size,tagset_size,embd_dim,hd_dim):
        super().__init__()
        self.input_dim = vocab_size
        self.output_dim = tagset_size
        self.hidden = hd_dim
        self.embd = embd_dim 
        # Creating an Embedding Layer
        self.embd_layer = nn.Embedding(self.input_dim,self.embd)
        # From Embedding Layer to Final Output Layer
        self.opt_predictor = nn.Sequential (nn.Linear(self.embd,self.hidden),                    # Embedding Layer to Hidden Layer
                                            nn.ReLU(),                                           # Activation Function
                                            nn.Linear(self.hidden,self.output_dim))              # Hidden Layer to Final Output
    def forward(self,idx):
        return self.opt_predictor(self.embd_layer(idx.to(torch.long)))

def ffnn_tagger():
    EMBD_DIM = 256
    HIDDEN_DIM = 256
    BATCH_SIZE = 120
    NUM_EPOCHS = 80
    LR = 0.2

    ffnn_model = FFNN_POS_Tagger(len(vocab),len(tags),EMBD_DIM,HIDDEN_DIM)

    ds = TensorDataset(X_train,Y_train)
    data = DataLoader(ds,batch_size=BATCH_SIZE,shuffle=True)

    Loss = nn.BCEWithLogitsLoss()
    Optimizer = optim.SGD(params=ffnn_model.parameters(),lr = LR)

    for i in range(0,NUM_EPOCHS):
        ffnn_model.train()
        for ipt_batch, opt_batch in data:
            opt_pred = ffnn_model(ipt_batch).squeeze()
            loss = Loss(opt_pred,opt_batch)
            Optimizer.zero_grad()
            loss.backward()
            Optimizer.step()
            
    return ffnn_model

##############################################################################################################################################################

def data2idx (data):
    X = []
    Y = []
    for sentence_data in data:
        word2idx = []
        tag2idx = []
        for i in range (0,len(sentence_data[0])):
            if sentence_data[1][i] in tags:
                tag2idx.append(tags[sentence_data[1][i]])
                if sentence_data[0][i] in vocab:
                    word2idx.append(vocab[sentence_data[0][i]])
                else :
                    word2idx.append(vocab['<unk>'])
        
        X.append(word2idx)
        y_vect = []
        for i in range (0,len(tag2idx)):
            y = [float(0)]*len(tags)
            y[tag2idx[i]] = float(1)
            y_vect.append(y)
        Y.append(y_vect)

    return X,Y

##############################################################################################################################################################

def padding (X,Y):
    # Maximum Sentence Size = 60
    sentence_size = 60
    y = [float(0)]*len(tags)
    y[tags['<pad>']] = float(1)
    for i in range (0,len(X)):
        sz = sentence_size - len(X[i])
        X[i] = X[i] + sz*[vocab['<pad>']]
        for j in range (0,sz):
            Y[i].append(y)
            j += 1

    ipt = torch.tensor(X)
    opt = torch.tensor(Y)

    return ipt,opt

##############################################################################################################################################################

def Evaluate_LSTM (lstm_model,BATCH_SIZE):
    # Loading the Test Data
    test_x,test_y = data2idx(dev_data)
    test_ipt,test_opt = padding(test_x,test_y)
    ds_test = TensorDataset(test_ipt,test_opt)
    test_loader = DataLoader(ds_test,batch_size=120,shuffle=True)

    pad = [float(0)]*len(tags)
    pad[tags['<pad>']] = float(1)
    pad_tensor = torch.tensor(pad)
    model_predictions = []
    actual_values = []
    correct = 0
    total = 0
    for ipt_test, opt_test in test_loader:
        lstm_model.initial_hd = (torch.zeros(1,min(BATCH_SIZE,len(ipt_test)),lstm_model.hidden),torch.zeros(1,min(BATCH_SIZE,len(ipt_test)),lstm_model.hidden))
        
        lstm_model.eval()
        with torch.inference_mode():
            pred_opt = torch.round(torch.sigmoid(lstm_model(ipt_test).squeeze()))

        for val in range (0,len(pred_opt)):
            for k in range (0,len(pred_opt[val])):
                if not torch.equal(opt_test[val][k],pad_tensor):
                    if torch.equal(opt_test[val][k],pred_opt[val][k]):
                        correct += 1
                        total += 1
                    else :
                        total += 1
                else:
                    model_predictions.append(pred_opt[val][:k].detach().numpy())
                    actual_values.append(opt_test[val][:k].detach().numpy())
                    break


    model_predictions = np.concatenate(model_predictions)
    actual_values = np.concatenate(actual_values)

    accuracy = correct/total
    _, recall_micro, f1_score_micro, _ = precision_recall_fscore_support(model_predictions, actual_values, average='micro',zero_division=1)
    _, recall_macro, f1_score_macro, _ = precision_recall_fscore_support(model_predictions, actual_values, average='macro',zero_division=1)

    print(f"Accuracy : {accuracy}")
    print("Micro-average Recall:", recall_micro)
    print("Micro-average F1-score:", f1_score_micro)
    print("Macro-average Recall:", recall_macro)
    print("Macro-average F1-score:", f1_score_macro)

    opt_labels = np.argmax(actual_values, axis=1)
    pred_labels = np.argmax(model_predictions, axis=1)
    
    confusion_mat = confusion_matrix(opt_labels,pred_labels)
    print("Confusion Matrix:")
    plt.figure(figsize=(15, 10))

    pos_vocab = []
    for tag in tags:
        pos_vocab.append(tag)

    # Displaying the confusion matrix as a heatmap
    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', xticklabels=pos_vocab, yticklabels=pos_vocab, annot_kws={'size': 12})
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

##############################################################################################################################################################

class LSTM_POS_Tagger (nn.Module):
    def __init__ (self,vocab_size,tagset_size,embd_dim,hd_dim,batch_size):
        super().__init__()
        self.input_dim = vocab_size
        self.output_dim = tagset_size
        self.hidden = hd_dim
        self.embd = embd_dim 
        self.batch_size = batch_size
        # Creating an Embedding Layer
        self.embd_layer = nn.Embedding(self.input_dim,self.embd)
        # From Embedding Layer to Final Output Layer
        self.lstm_layer = nn.LSTM(self.embd,self.hidden,batch_first=True)
        self.hidden_layer = nn.Linear(self.hidden,self.output_dim)

        # dimensions (no. of lstm layer, batch size 1 (processing one sentence at a time), hidden_layer dimensions)
        # initializing the hidden layer to zero for first iteration
        self.initial_hd = (torch.zeros(1,self.batch_size,self.hidden),torch.zeros(1,self.batch_size,self.hidden))

    def forward(self,batch):
        embd_layer = self.embd_layer(batch)
        lstm_layer, self.initial_hiddenlayer = self.lstm_layer(embd_layer,self.initial_hd)
        tag_layer = self.hidden_layer(lstm_layer)

        return tag_layer

def lstm_tagger():
    EMBD_DIM = 128
    HIDDEN_DIM = 128
    BATCH_SIZE = 120
    NUM_EPOCHS = 30
    LR = 0.1

    X,Y = data2idx(train_data)
    ipt,opt = padding(X,Y)

    lstm_model = LSTM_POS_Tagger(len(vocab),len(tags),EMBD_DIM,HIDDEN_DIM,BATCH_SIZE)

    ds = TensorDataset(ipt,opt)
    data = DataLoader(ds,batch_size=BATCH_SIZE,shuffle=True)
    
    Loss = nn.BCEWithLogitsLoss()
    Optimizer = optim.Adam(params=lstm_model.parameters(),lr = LR)

    for i in range(0,NUM_EPOCHS):
        for ipt_batch, opt_batch in data:
            lstm_model.zero_grad()
            lstm_model.initial_hd = (torch.zeros(1,min(BATCH_SIZE,len(ipt_batch)),lstm_model.hidden),torch.zeros(1,min(BATCH_SIZE,len(ipt_batch)),lstm_model.hidden))
            pred_tags = lstm_model(ipt_batch)
            loss = Loss(pred_tags,opt_batch)
            loss.backward()
            Optimizer.step()

    return lstm_model

##############################################################################################################################################################

def pos_tags_sentence(model,sentence):
    input_sent = remove_punctuation(sentence).lower().split()

    sent_idx = []
    for word in input_sent:
        if word in vocab:
            sent_idx.append(vocab[word])
        else:
            sent_idx.append(vocab['<unk>'])

    sent_idx = sent_idx + [vocab['<pad>']]*(60-len(input_sent))
    sent_tensor = torch.tensor(sent_idx).view(1,60)

    model.initial_hd = (torch.zeros(1,len(sent_tensor),model.hidden),torch.zeros(1,len(sent_tensor),model.hidden))
    with torch.inference_mode():
        pred_opt = torch.round(torch.sigmoid(model(sent_tensor).squeeze()))

    for i in range(0,len(input_sent)):
        flag = 0
        for j in range (0,len(pred_opt[i])):
            if int(pred_opt[i][j]) == 1:
                flag = 1
                print (f"{input_sent[i]} {idx_tags[j]}")
                break
        if flag == 0:
            print (f"{input_sent[i]} NOUN")

##############################################################################################################################################################
            
def graph(x_label,y_label):
    plt.plot(x_label, y_label, marker='o')

    plt.xlabel('Context Window Size')
    plt.ylabel('Accuracy')
    plt.title('Context Window Size vs Accuracy')

    plt.grid(True)
    plt.show()

##############################################################################################################################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', action='store_true', help='Feed Forward Neural Network')
    parser.add_argument('-r', action='store_true', help='Recurrent Neural Network')
    args = parser.parse_args()

    input_sentence = input("Enter a sentence: ")
    
    if args.f and not args.r:
        """ Training the Model and Saving it to .pt file """
        # ff_model = ffnn_tagger()
        # torch.save(ff_model.state_dict(),'ffnn_model.pt')

        """ Loading the Model from .pt file and using it """
        ff_model = FFNN_POS_Tagger(len(vocab),len(tags),256,256)
        ff_model.load_state_dict(torch.load('ffnn_model.pt'))
        Model_Evaluate(ff_model,X_test,Y_test)
        tag_sentence(ff_model,input_sentence)
        
    elif args.r and not args.f:
        """ Training the Model and Saving it to .pt file """
        # lstm_model = lstm_tagger()
        # torch.save(lstm_model.state_dict(),'lstm_model.pt')

        """ Loading the Model from .pt file and using it """
        lstm_model = LSTM_POS_Tagger(len(vocab),len(tags),128,128,120)
        lstm_model.load_state_dict(torch.load('lstm_model.pt'))
        Evaluate_LSTM(lstm_model,120)
        pos_tags_sentence(lstm_model,input_sentence)


    else:
        print("Please choose model type between (-f) Feed Forward Neural Network and (-r) Recurrent Neural Network")

if __name__ == "__main__":
    main()
