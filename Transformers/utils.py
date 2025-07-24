import math
import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

MODEL_DIM = 512
DROPOUT = 0.2
SEQ_LEN = 45
FFD = 1024
NUM_HEADS = 8   # MODEL_DIM must be divisible by NUM_HEADS so that we can divided the matrix along the embedding dimension during Multihead training
NUM_BLOCKS = 6
LR = 0.001
EPOCHS = 5
BATCH_SIZE = 64
    
class LayerNorm (nn.Module):
    def __init__(self, eps: float = 10**-6):
        super(LayerNorm,self).__init__()
        self.alpha = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(1))
        self.eps = eps

    def forward (self, x):
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim = -1, keepdim=True)
        output = self.alpha*(x - mean)/std+self.eps + self.bias
        return output
    
class PositionalEncoding(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(DROPOUT)
        pe = torch.zeros(SEQ_LEN, MODEL_DIM)
        position = torch.arange(0, SEQ_LEN, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        div_term = torch.exp(torch.arange(0, MODEL_DIM, 2).float() * (-math.log(10000.0) / MODEL_DIM)) # (model_dim / 2)
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / model_dim))
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / model_dim))
        pe = pe.unsqueeze(0) # (1, seq_len, model_dim)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, model_dim)
        return self.dropout(x)

class FeedForward (nn.Module):
    def __init__(self):
        super(FeedForward,self).__init__()
        self.linear = nn.Linear(MODEL_DIM,FFD)
        self.dropout = nn.Dropout(DROPOUT)
        self.ffd_output = nn.Linear(FFD,MODEL_DIM)

    def forward (self,x):
        l1 = self.dropout(torch.relu(self.linear(x)))
        return self.ffd_output(l1)
    
class MultiHeadAttention (nn.Module):
    def __init__(self):
        super(MultiHeadAttention,self).__init__()
        self.wq = nn.Linear(MODEL_DIM,MODEL_DIM,bias=False)
        self.wk = nn.Linear(MODEL_DIM,MODEL_DIM,bias=False)
        self.wv = nn.Linear(MODEL_DIM,MODEL_DIM,bias=False)
        self.wo = nn.Linear(MODEL_DIM,MODEL_DIM,bias=False)
        self.dk = MODEL_DIM//NUM_HEADS
        self.dropout = nn.Dropout(DROPOUT)

    @staticmethod
    def self_attention(query, key, value, mask):
        scores = (query @ key.transpose(-2,-1))/math.sqrt(MODEL_DIM/NUM_HEADS)    
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)
        scores = scores.softmax(dim = -1)    #(batch, num_heads, seq_len, seq_len)
        return (scores @ value), scores

    def forward (self,q,k,v,mask):
        # print(f"q in attention: {q}")  
        # print(f"k in attention: {k}") 
        # print(f"v in attention: {v}")
        query = self.wq(q)    # (batch, seq_len, model_dim)
        key = self.wk(k)
        value = self.wv(v)
        # dividing the matrices into smaller matrices for multihead training
        query = query.view(query.shape[0], query.shape[1], NUM_HEADS, self.dk).transpose(1,2)   #(batch, num_heads, seq_lenght, dk)
        key = key.view(key.shape[0], key.shape[1], NUM_HEADS, self.dk).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], NUM_HEADS, self.dk).transpose(1,2)
        x, _ = MultiHeadAttention.self_attention(query,key,value,mask)

        # x - batch, num_heads, seq_len, dk -> batch, seq_len, num_heads, dk
        x = x.transpose(1,2)
        x = x.reshape(x.shape[0], -1, MODEL_DIM)
        return self.wo(x)

class ResidualConnection (nn.Module):
    def __init__(self):
        super(ResidualConnection,self).__init__()
        self.norm = LayerNorm()
        self.dropout = nn.Dropout(DROPOUT)

    def forward (self, x, prev_layer):
        # print(f"x before norm in residual connection: {x}")
        return x + self.dropout(prev_layer(self.norm(x)))

class Embeddings (nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.get_embeddings = nn.Embedding(vocab_size,MODEL_DIM)

    def forward(self,x):
        return self.get_embeddings(x)*math.sqrt(MODEL_DIM)

class Transformer (nn.Module):
    def __init__ (self,encoder,decoder,pos,vocab_fr,input_embd,output_embd,Projection_Layer):
        super(Transformer,self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.output_size = vocab_fr
        self.positional = pos
        self.input_embd = input_embd
        self.output_embd = output_embd
        self.projection = Projection_Layer

    def encode (self, ipt, mask):
        # print(f"ipt before embedding: {ipt}")
        ipt = self.input_embd(ipt)
        final_input = self.positional(ipt)
        return self.encoder(final_input,mask)
    
    def decode (self,encd_opt,mask,opt,tgt_mask):
        opt = self.output_embd(opt)
        final_output = self.positional(opt)
        return self.decoder(final_output,encd_opt,mask,tgt_mask)

    def final_projection(self,decd_opt):
        return self.projection(decd_opt)
