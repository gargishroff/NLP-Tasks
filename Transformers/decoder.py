import utils
import torch
import torch.nn as nn

class Decoder (nn.Module):
    def __init__(self, self_attention_layer:utils.MultiHeadAttention, cross_attention_layer:utils.MultiHeadAttention, feed_forward_layer:utils.FeedForward):
        super(Decoder,self).__init__()
        self.attention_layer = self_attention_layer
        self.feed_forward_layer = feed_forward_layer
        self.cross_attention = cross_attention_layer
        self.rc1 = utils.ResidualConnection()
        self.rc2 = utils.ResidualConnection()
        self.rc3 = utils.ResidualConnection()

    def forward (self, x, encd, mask, tgt_mask):
        x = self.rc1(x, lambda x : self.attention_layer(x,x,x,tgt_mask))
        x = self.rc2(x, lambda x: self.cross_attention(x, encd, encd, mask))
        x = self.rc3(x, self.feed_forward_layer)
        return x

class CombinedDecoder (nn.Module):
    def __init__(self, blocks:nn.ModuleList):
        super(CombinedDecoder,self).__init__()
        self.blocks = blocks
        self.norm = utils.LayerNorm()

    def forward (self,x,encd,mask,tgt_mask):
        for decoder_block in self.blocks:
            x = decoder_block(x,encd,mask,tgt_mask)
        return self.norm(x)
    
class ProjectOutputs(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.project = nn.Linear(utils.MODEL_DIM,output_size)

    def forward(self,x):
        return self.project(x)
