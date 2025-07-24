import re
import torch 
import pickle
import numpy as np
from torch import nn,optim
from torch.utils.data import DataLoader, TensorDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def Identify_URLS(tokenized_words):
    pattern = r'^http|www'
    for i, sentence_words in enumerate(tokenized_words):
        for j, word in enumerate(sentence_words):
            if re.match(pattern, word):
                tokenized_words[i][j] = '<URL>'
    return Rem_Punct(tokenized_words)

def Identify_Mails(tokenized_words):
    pattern = r'\w[^\s{}()\[\]]*@[^\s{}()\[\]]*\w'
    for i, sentence_words in enumerate(tokenized_words):
        for j, word in enumerate(sentence_words):
            if re.match(pattern, word):
                tokenized_words[i][j] = '<MAILID>'
    return Identify_URLS(tokenized_words)

def Identify_Mentions(tokenized_words):
    pattern = r'@[^\s{}()\[\]]*\w'
    for i, sentence_words in enumerate(tokenized_words):
        for j, word in enumerate(sentence_words):
            if re.match(pattern, word):
                tokenized_words[i][j] = '<MENTION>'
    return Identify_Mails(tokenized_words)

def Identify_Hashtags(tokenized_words):
    pattern = r'#[^\s{}()\[\]]*\w'
    for i, sentence_words in enumerate(tokenized_words):
        for j, word in enumerate(sentence_words):
            if re.match(pattern, word):
                tokenized_words[i][j] = '<HASHTAG>'
    return Identify_Mentions(tokenized_words)

def Identify_Num (tokenized_words):
    pattern = r'\d+\.\d+(?=\s|$)|\d[\d+,]*(?=\s|$)|\d+(?=\s|$)'
    for i, sentence_words in enumerate(tokenized_words):
        for j, word in enumerate(sentence_words):
            if re.match(pattern, word):
                tokenized_words[i][j] = '<NUM>'
    return Identify_Hashtags(tokenized_words)

def Rem_Punct(tokenized_words):
    filtered_words = []
    for sentence_words in tokenized_words:
        list = []
        for word in sentence_words:
            if len(word) == 1:
                if re.match(r'^[^\w(){}\[\]]', word):
                    continue
                else:
                    list.append(word)
            else:
                list.append(word)
        if list:
            filtered_words.append(list)
    return filtered_words

def words_tokenize(sentences):
    tokenized_words = []
    for sentence in sentences:
        sent = sentence
        words = re.findall(r"'s|@[^\s{}()\[\]]*\w|#[^\s{}()\[\]]*\w|\w[^'\s{}()\[\]]*\w|\w+|\S",sent)
        tokenized_words.append(words)
    return Identify_Num(tokenized_words)

def sentence_tokenize(doc):
    pattern = re.compile(r'(?<![A-Z]\.|[0-9]\.)(?<!Mr\.|Dr\.)(?<!Mrs\.)(?<!\.\.)(?<=\.|\?|\!)\s' + '|' + r'(?<=\.\"|\?\"|\!\")\s' + '|' + r'\n\n+')
    ignore = r'\n'
    snts = pattern.split(doc)
    sentences = []
    for snt in snts:
        sentence = re.sub(ignore,' ',snt)
        if sentence:
            sentences.append(sentence)
    return words_tokenize(sentences)

def tokenizer(doc):
    return sentence_tokenize(doc)

def tokenize_file(input_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        text = file.read()

    tokenized_text = tokenizer(text)
    return tokenized_text

###################################################################################################################################

def load_glove_embeddings(glove_file):
    embeddings_index = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            embedding = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = embedding
    return embeddings_index

###################################################################################################################################

def generate_word2idx (tokenized_data):
    word2idx = {}
    i = 0
    for sentence in tokenized_data:
        for word in sentence:
            if word not in word2idx:
                word2idx[word] = i
                i = i + 1
    
    word2idx['<unk>'] = len(word2idx)
    return word2idx

###################################################################################################################################

def generate_5_grams (tokenized_data):
    # ngrams - list (complete data) -> list (sentence wise) -> tuple of 5 grams 
    # next_word - list (complete data) -> list (sentence wise) -> next word corresponding to that tuple
    ngrams = []
    next_word = []
    for sentence in tokenized_data:
        sentence_ngrams = []
        sentence_next_word = []
        sentence = 5*['<s>'] + sentence
        for i in range (len(sentence) - 5):
            ng = tuple(sentence[i:i+5])
            sentence_ngrams.append(ng)
            sentence_next_word.append(sentence[i+5])
        ngrams.append(sentence_ngrams)
        next_word.append(sentence_next_word)
    return ngrams, next_word

###################################################################################################################################

def create_embedding_matrix(word2idx, embeddings_index):
    vocab_size = len(word2idx)
    embedding_matrix = np.zeros((vocab_size, 100))

    for word, idx in word2idx.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Words not found in the embedding index will be all zeros
            embedding_matrix[idx] = embedding_vector
        else:
            # Random initialization for out-of-vocabulary words
            embedding_matrix[idx] = np.random.normal(scale=0.6, size=(100,))
    return torch.tensor(embedding_matrix, dtype=torch.float).to(device)

###################################################################################################################################

def generate_model_data(ngrams,next_word,embedding_matrix,word2idx):
    embedded_ngrams = []
    embedded_targets = []

    for sentence_ngrams, sentence_next_words in zip(ngrams, next_word):
        sentence_embedded_ngrams = []
        sentence_embedded_targets = []

        for ng, target in zip(sentence_ngrams, sentence_next_words):
            # Convert n-gram words to their embedding vectors
            ng_embeds = [embedding_matrix[word2idx.get(word, word2idx['<unk>'])] for word in ng]
            target_embed = word2idx.get(target, word2idx['<unk>'])

            # Concatenate the n-gram embeddings (5-grams)
            sentence_embedded_ngrams.append(torch.cat(ng_embeds))
            sentence_embedded_targets.append(target_embed)

        # embedded_ngrams.append(torch.stack(sentence_embedded_ngrams))
        # embedded_targets.append(torch.stack(sentence_embedded_targets))
        embedded_ngrams.extend(sentence_embedded_ngrams)
        embedded_targets.extend(sentence_embedded_targets)

    return torch.stack(embedded_ngrams).to(device), torch.tensor(embedded_targets,dtype=torch.long).to(device)

###################################################################################################################################

HIDDEN_DIM = 300
EMDB_DIM = 500    # 100 Dimension Embedding from Glove of 5 words concatenated together
LR = 0.001
BATCH_SIZE = 100
EPOCHS = 6

class NeuralLanguageModel (nn.Module):
    def __init__ (self,vocab_size):
        super(NeuralLanguageModel,self).__init__()
        self.hidden_layer1 = nn.Linear(EMDB_DIM,HIDDEN_DIM)
        self.dropout = nn.Dropout(p=0.3)
        self.hidden_layer2 = nn.Linear(HIDDEN_DIM,vocab_size)
        # self.softmax = nn.Softmax(dim=1)

    def forward (self,x):
        hidden_layer1 = nn.functional.relu(self.hidden_layer1(x))
        hidden_layer1 = self.dropout(hidden_layer1)
        hidden_layer2 = self.hidden_layer2(hidden_layer1)
        return hidden_layer2

def training_loop(vocab_size,data,val_data,test_data):
    model = NeuralLanguageModel(vocab_size).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),weight_decay=1e-5,lr=LR)

    model.train()
    for epoch in range(EPOCHS):
        tloss = 0
        for ngrams_batch, targets_batch in data:
            ngrams_batch, targets_batch = ngrams_batch.to(device), targets_batch.to(device)

            # Forward pass
            outputs = model(ngrams_batch)
            loss = loss_function(outputs, targets_batch)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tloss += loss.item()
        print(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {tloss/len(data):.4f}")

        with torch.no_grad():
            log_probs_val = []
            for ngrams_batch_val, targets_batch_val in val_data:
                ngrams_batch_val, targets_batch_val = ngrams_batch_val.to(device), targets_batch_val.to(device)
                outputs = model(ngrams_batch_val)
                loss_val = loss_function(outputs,targets_batch_val)
                log_probs_val.append(loss_val)

        mean_loss_val = torch.mean(torch.tensor(log_probs_val))
        print(f"Epoch [{epoch+1}/{EPOCHS}], Validation Loss: {mean_loss_val:.4f}")
        perplexity_val = torch.exp(mean_loss_val)
        print(f"Perplexity: {perplexity_val.item()}")
        print()

    torch.save(model.state_dict(),'NeuralLanguageModel.pt')

    model.eval()
    with torch.no_grad():
        log_probs_test = []
        for ngrams_batch_test, targets_batch_test in test_data:
            ngrams_batch_test, targets_batch_test = ngrams_batch_test.to(device), targets_batch_test.to(device)
            outputs = model(ngrams_batch_test)
            loss = loss_function(outputs,targets_batch_test)
            log_probs_test.append(loss)

    mean_loss_test = torch.mean(torch.tensor(log_probs_test))
    perplexity_test = torch.exp(mean_loss_test)
    print(f"Perplexity on Test Data: {perplexity_test.item()}")

###################################################################################################################################

def sentence_5grams (sentence,embedding_matrix,word2idx):
    sentence_ngrams = []
    sentence_next_word = []
    sentence = 5*['<s>'] + sentence
    for i in range (len(sentence) - 5):
        ng = tuple(sentence[i:i+5])
        sentence_ngrams.append(ng)
        sentence_next_word.append(sentence[i+5])

    sentence_embedded_ngrams = []
    sentence_embedded_targets = []
    for ng, target in zip(sentence_ngrams, sentence_next_word):
        ng_embeds = [embedding_matrix[word2idx.get(word, word2idx['<unk>'])] for word in ng]
        target_embed = word2idx.get(target, word2idx['<unk>'])
        sentence_embedded_ngrams.append(torch.cat(ng_embeds))
        sentence_embedded_targets.append(target_embed)

    return torch.stack(sentence_embedded_ngrams).to(device), torch.tensor(sentence_embedded_targets,dtype=torch.long).to(device)

def CalculatePerplexity(model,data,embedding_matrix,word2idx):
    model.eval()
    loss_function = nn.CrossEntropyLoss()

    with open("2022114009-LM1-val-perplexity.txt", 'w') as file:
        log_probs = []
        for sentence in data:
            X, Y = sentence_5grams(sentence, embedding_matrix, word2idx)
            with torch.no_grad():
                X, Y = X.to(device), Y.to(device)
                outputs = model(X)
                loss = loss_function(outputs, Y)
                perplexity = torch.exp(loss)
                log_probs.append(loss)

                file.write(f"{' '.join(sentence)}\t{perplexity.item():.4f}\n")

        mean_loss = torch.mean(torch.tensor(log_probs))
        perplexity_avg = torch.exp(mean_loss)
        file.write(f"\nAverage Perplexity:\t{perplexity_avg.item():.4f}\n")

        print(f"Average Perplexity:\t{perplexity_avg.item():.4f}")

###################################################################################################################################

def main ():
    # tokenized_data = tokenize_file("dataset/train.txt")
    # with open('tokenized_data_train.pkl', 'wb') as file:
    #     pickle.dump(tokenized_data, file)

    with open('tokenized_data_train.pkl', 'rb') as file:
        tokenized_data = pickle.load(file)
        
    # embeddings_index = load_glove_embeddings("glove.6B.100d.txt")
    # with open('glove.pkl', 'wb') as file:
    #     pickle.dump(embeddings_index, file)
    with open('glove.pkl', 'rb') as file:
        embeddings_index = pickle.load(file)

    # word2idx = generate_word2idx(tokenized_data)
    # with open('word2idx.pkl', 'wb') as file:
    #     pickle.dump(word2idx, file)
    with open('word2idx.pkl', 'rb') as file:
        word2idx = pickle.load(file)

    embedding_matrix = create_embedding_matrix(word2idx,embeddings_index)
    vocab_size = len(word2idx)

    # ngrams,next_word = generate_5_grams(tokenized_data)
    # embedded_ngrams, embedded_targets = generate_model_data(ngrams,next_word,embedding_matrix,word2idx)
    # dataset = TensorDataset(embedded_ngrams,embedded_targets)
    # data = DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True)

    # tokenized_val_data = tokenize_file("dataset/val.txt")
    # val_ngrams, val_nextword = generate_5_grams(tokenized_val_data)
    # embedded_ngrams_val, embedded_targets_val = generate_model_data(val_ngrams,val_nextword,embedding_matrix,word2idx)
    # val_dataset = TensorDataset(embedded_ngrams_val,embedded_targets_val)
    # val_data = DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=True)

    # tokenized_test_data = tokenize_file("dataset/test.txt")
    # test_ngrams, test_nextword = generate_5_grams(tokenized_test_data)
    # embedded_ngrams_test, embedded_targets_test = generate_model_data(test_ngrams,test_nextword,embedding_matrix,word2idx)
    # test_dataset = TensorDataset(embedded_ngrams_test,embedded_targets_test)
    # test_data = DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False)

    # training_loop(vocab_size, data,val_data,test_data)
    
    NeuralLM = NeuralLanguageModel(vocab_size).to(device)
    NeuralLM.load_state_dict(torch.load('NeuralLanguageModel.pt'))
    NeuralLM.eval()
    data = tokenize_file("dataset/val.txt")
    CalculatePerplexity(NeuralLM,data,embedding_matrix,word2idx)

if __name__ == "__main__":
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    np.random.seed(42)
    main ()