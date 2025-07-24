import re
import math
import torch 
import pickle
import numpy as np
from torch import nn,optim
from torch.utils.data import DataLoader, TensorDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

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

def generate_model_data(tokenized_data,word2idx):
    """
    data_embeddings contains words indices of the sentence (60 words at max) - list(sentences) -> list(words indices)
    target_indices contains word indices for the next word to be predicted - list(sentences) -> list(words indices)
    """
    max_sentence_length = 60
    data_embeddings = []
    target_indices = []

    for sentence in tokenized_data:
        len_sent = 0
        sentence_embeddings = []
        sentence_indices = []

        for word in sentence:
            sentence_embeddings.append(word2idx.get(word, word2idx['<unk>']))
            if len_sent + 1 < len(sentence):  
                sentence_indices.append(word2idx.get(sentence[len_sent + 1], word2idx['<unk>']))
            else:
                sentence_indices.append(word2idx.get('<pad>', word2idx['<unk>']))
            len_sent += 1
            if len_sent == max_sentence_length:
                break

        while len_sent < max_sentence_length:
            sentence_embeddings.append(word2idx['<pad>'])  # Padding embedding
            sentence_indices.append(word2idx['<pad>'])  # Padding index
            len_sent += 1

        data_embeddings.append(sentence_embeddings)
        target_indices.append(sentence_indices)

    data_embeddings = torch.tensor(data_embeddings, dtype=torch.long).to(device)
    target_indices = torch.tensor(target_indices, dtype=torch.long).to(device)

    return data_embeddings, target_indices

###################################################################################################################################

BATCH_SIZE = 128
MAX_LEN = 60
LAYERS = 5
HEADS = 5
FFN_DIM = 300
EMBD_DIM = 100
EPOCHS = 20
LR = 0.01

class TransformerLanguageModel (nn.Module):
    def __init__(self,vocab_size,embedding_matrix):
        super(TransformerLanguageModel,self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        self.positional = self.positional_encoding()
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=EMBD_DIM,nhead=HEADS,dim_feedforward=FFN_DIM)
        self.decoder = nn.TransformerDecoder(self.decoder_layer,num_layers=LAYERS)
        self.final_layer = nn.Linear(EMBD_DIM,vocab_size)

    def positional_encoding(self):
        pe = torch.zeros(MAX_LEN,EMBD_DIM)
        position = torch.arange(0,MAX_LEN,dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, EMBD_DIM, 2).float() * (-math.log(10000.0) / EMBD_DIM))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
    
    def generate_lookahead_mask(self):
        mask = torch.triu(torch.ones(MAX_LEN,MAX_LEN)* float('-inf'), diagonal=1)
        return mask
    
    def forward(self,input_seq,output_seq):
        batch_size = input_seq.size(0)
        mask = self.generate_lookahead_mask().unsqueeze(0).repeat(batch_size*HEADS,1,1).to(device)
        embedded_input = self.embedding(input_seq) * math.sqrt(EMBD_DIM)
        embedded_target = self.embedding(output_seq) * math.sqrt(EMBD_DIM)

        pos = self.positional.unsqueeze(0).repeat(batch_size,1,1).to(device)
        encoded_input = embedded_input+pos[:,:MAX_LEN,:]
        encoded_output = embedded_target+pos[:,:MAX_LEN,:]

        encoded_input = encoded_input.transpose(0,1)
        encoded_output = encoded_output.transpose(0,1)

        decoded_output = self.decoder(tgt=encoded_output,memory=encoded_input,tgt_mask=mask)
        decoded_output = decoded_output.transpose(0,1)
        output = self.final_layer(decoded_output)
        return output

def training_loop(word2idx,embedding_matrix,tokenized_data):
    data_embeddings, target_indices = generate_model_data(tokenized_data,word2idx)
    vocab_size = len(word2idx)

    ds = TensorDataset(data_embeddings,target_indices)
    data = DataLoader(ds,batch_size=BATCH_SIZE,shuffle=True)

    transformer_model = TransformerLanguageModel(vocab_size,embedding_matrix).to(device)

    optimizer = optim.Adam(transformer_model.parameters(),lr = LR,weight_decay=1e-3)
    loss_fn = nn.CrossEntropyLoss(ignore_index=word2idx['<pad>'])

    for epoch in range(EPOCHS):
        total_loss = 0
        transformer_model.train()

        for batch_idx, (batch_data, batch_target) in enumerate(data):
            optimizer.zero_grad()
            output = transformer_model(batch_data,batch_target)

            output = output.view(-1, vocab_size)  # Flatten the output to (batch_size * seq_length, vocab_size)
            batch_target = batch_target.view(-1) 

            loss = loss_fn(output,batch_target)
            total_loss += loss

            loss.backward()
            optimizer.step()

        avg_loss = total_loss/len(data)
        print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}')

    return transformer_model

###################################################################################################################################

def sentence_data (sentence,word2idx):
    max_sentence_length = 60
    len_sent = 0
    sentence_embeddings = []
    sentence_indices = []
    for word in sentence:
        sentence_embeddings.append(word2idx.get(word, word2idx['<unk>']))
        if len_sent + 1 < len(sentence):  
            sentence_indices.append(word2idx.get(sentence[len_sent + 1], word2idx['<unk>']))
        else:
            sentence_indices.append(word2idx.get('<pad>', word2idx['<unk>']))
        len_sent += 1
        if len_sent == max_sentence_length:
            break

    while len_sent < max_sentence_length:
        sentence_embeddings.append(word2idx['<pad>'])  # Padding embedding
        sentence_indices.append(word2idx['<pad>'])  # Padding index
        len_sent += 1

    return sentence_embeddings,sentence_indices

def CalculatePerplexity (model,data,word2idx):
    model.eval()
    loss_function = nn.CrossEntropyLoss(ignore_index=word2idx['<pad>'])
    log_probs = []
    with open("2022114009-LM3-val-perplexity.txt", 'w') as file:
        for sentence in data:
            if (len(sentence) <= 1):
                continue
            X,Y = sentence_data(sentence,word2idx)
            X = torch.tensor([X], dtype=torch.long).to(device) 
            Y = torch.tensor([Y], dtype=torch.long).to(device)

            with torch.no_grad():  
                output = model(X,Y)  
                output = output.view(-1, len(word2idx))  # Shape (max_sentence_length, vocab_size)
                Y = Y.view(-1) 
                loss = loss_function(output, Y)
                perplexity = torch.exp(loss)
                log_probs.append(loss)
                
                file.write(f"{' '.join(sentence)}\t{perplexity.item():.4f}\n")
            
        mean_loss = torch.mean(torch.tensor(log_probs))
        perplexity_avg = torch.exp(mean_loss)
        file.write(f"\nAverage Perplexity:\t{perplexity_avg.item():.4f}\n")
    print(f"Average Perplexity: {perplexity_avg.item():.4f}")

###################################################################################################################################

def main ():
    with open('tokenized_data_train.pkl', 'rb') as file:
        tokenized_data = pickle.load(file)

    with open('glove.pkl', 'rb') as file:
        embeddings_index = pickle.load(file)

    with open('word2idx.pkl', 'rb') as file:
        word2idx = pickle.load(file)

    word2idx['<pad>'] = len(word2idx)
    vocab_size = len(word2idx)
    embedding_matrix = create_embedding_matrix(word2idx,embeddings_index)
    # transformer_model = training_loop(word2idx,embedding_matrix,tokenized_data)
    # torch.save(transformer_model.state_dict(),'transformer_model.pt')

    data = tokenize_file("dataset/val.txt")
    transformer_model = TransformerLanguageModel(vocab_size,embedding_matrix).to(device)
    transformer_model.load_state_dict(torch.load('transformer_model.pt'))
    transformer_model.eval()
    CalculatePerplexity(transformer_model,data,word2idx)


if __name__ == "__main__":
    main()