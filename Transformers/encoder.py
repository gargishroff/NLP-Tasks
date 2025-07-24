import utils
import torch
import torch.nn as nn

class Encoder (nn.Module):
    def __init__(self, self_attention_layer:utils.MultiHeadAttention, feed_forward_layer:utils.FeedForward):
        super(Encoder,self).__init__()
        self.attention_layer = self_attention_layer
        self.feed_forward_layer = feed_forward_layer
        self.rc1 = utils.ResidualConnection()
        self.rc2 = utils.ResidualConnection()

    def forward(self, x, mask):
        x = self.rc1(x, lambda x : self.attention_layer(x,x,x,mask))
        # print(f"x after rc1: {x}")
        x = self.rc2(x, self.feed_forward_layer)
        return x

class CombinedEncoder (nn.Module):
    def __init__(self, blocks : nn.ModuleList):
        super(CombinedEncoder,self).__init__()
        self.blocks = blocks
        self.norm = utils.LayerNorm()

    def forward (self,x,mask):
        for encoder_block in self.blocks:
            x = encoder_block(x,mask)
        return self.norm(x)
