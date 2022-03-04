from dataclasses import replace
from tkinter.tix import Tree
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import math


## 6. MultiHeadAttention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_k, d_v, d_model, d_ff, n_heads=4, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        # read config
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.d_ff = d_ff
        self.n_heads = n_heads

     
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias = False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias = False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias = False)

        # optional 
        self.linear = nn.Linear(n_heads * d_v, d_model) # for output projection
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # for FFN
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.FFN_Relu = nn.ReLU(inplace=True)
        self.FFN_dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)

        # another FFN version implemented with nn.Linear Layer
        self.FFN = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout), # optional
            nn.Linear(d_ff, d_model, bias=True)
        )        
       
        # self.dropout2 = nn.Dropout(dropout)
        self.layer_norm_ffn = nn.LayerNorm(d_model)

        def init_weight(model):
            for m in model.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            

        # init_weight(self)



    def forward(self, x):# x: [batch_size x len_q x d_model]

        residual, batch_size = x, x.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)

        Q = self.W_Q(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
        K = self.W_K(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        V = self.W_V(x).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]


        #context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q x len_k]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)

        attn = torch.softmax(scores,dim=-1)
        context = torch.matmul(attn, V)

        output = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v) # context: [batch_size x len_q x n_heads * d_v]
        # output = self.linear(output) #optional
        output = self.layer_norm(self.dropout1(output) + residual) # self.dropout1(output) is optional
        # output: [batch_size x len_q x d_model]

        residual = output # output : [batch_size, len_q, d_model]

        # one version of FFN
        output = self.FFN_Relu(self.conv1(output.transpose(1, 2)))
        # output = self.FFN_dropout(output) # optional
        output = self.conv2(output).transpose(1, 2)

        # another version of FFN
        # output = self.FFN(output)

        output = self.layer_norm_ffn(output + residual)
        # output : [batch_size, len_q, d_model]
        
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len=100, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.d_model=d_model
        self.seq_len=seq_len
        self.dropout = nn.Dropout(p=dropout)
        # self.register_buffer('pe', self.pe)  ## 定一个缓冲区，其实简单理解为这个参数不更新就可以

    def forward(self, x):
        """
        x: [batch_size, d_model, seq_len]
        """
        x = x.transpose(1, 2)
        pe = torch.zeros(self.seq_len, self.d_model)
        position = torch.arange(0, self.seq_len, dtype=torch.float).unsqueeze(1)#(max_len, 1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)## 这里需要注意的是pe[:, 0::2]这个用法，就是从0开始到最后面，补长为2，其实代表的就是偶数位置
        pe[:, 1::2] = torch.cos(position * div_term)##这里需要注意的是pe[:, 1::2]这个用法，就是从1开始到最后面，补长为2，其实代表的就是奇数位置
        # pe:[max_len*d_model]
    
        pe = pe.unsqueeze(0).to(x.device) 
        x = x + pe #[batch_size, seq_len, d_model]
        
        return  x #self.dropout(x) # x


class Encoder(nn.Module):#d_k=d_v d_model=256
    def __init__(self, d_model, seq_len, n_layers, d_k, d_v, d_ff, n_heads=4, dropout=0.1):
        super(Encoder, self).__init__()
        self.pos_emb = PositionalEncoding(d_model,seq_len,dropout) ## 位置编码情况，这里是固定的正余弦函数，也可以使用类似词向量的nn.Embedding获得一个可以更新学习的位置编码
        # d_k, d_v, d_model, d_ff, n_heads=4
        self.layers = nn.ModuleList([MultiHeadAttention(d_k, d_v, d_model, d_ff, n_heads) for _ in range(n_layers)]) ## 使用ModuleList对多个encoder进行堆叠，因为后续的encoder并没有使用词向量和位置编码，所以抽离出来；

    def forward(self, x):
        '''
        x: [batch_size, d_model, seq_len]
        '''
        x = self.pos_emb(x)

        for layer in self.layers:
            x = layer(x)

        x = x.transpose(1, 2)  
        return x

    

class Transformer(nn.Module):
    def __init__(self, in_dim, dropout = 0.1):
        super(Transformer, self).__init__()
        self.in_dim = in_dim
        self.Q_W = nn.Linear(in_dim, in_dim, bias = False)
        self.K_W = nn.Linear(in_dim, in_dim, bias = False)
        self.V_W = nn.Linear(in_dim, in_dim, bias = False)
        self.layernorm1 = nn.LayerNorm([in_dim])
        self.dropout1 = nn.Dropout(dropout)

        self.FFN_linear1 = nn.Linear(in_dim, in_dim, bias = True)
        self.FFN_relu = nn.ReLU(inplace=True)
        self.FFN_dropout = nn.Dropout(dropout)
        self.FFN_linear2 = nn.Linear(in_dim, in_dim, bias=True)

        self.dropout2 = nn.Dropout(dropout)
        self.layernorm2 = nn.LayerNorm([in_dim])

    def forward(self, x):
        '''
        :param x: shape [B, dim, len]
        :return:
        '''    
        x = x.transpose(1,2) #[b,len,dim]

        query = self.Q_W(x)
        keys = self.K_W(x).transpose(1, 2)
        values = self.V_W(x)
        att_weight = torch.bmm(query, keys) / math.sqrt(self.in_dim)
        # att_weight = torch.einsum('bnc,bcm->bnm', query, keys) / math.sqrt(self.in_dim)
        att_weight = torch.softmax(att_weight, dim = -1)
        att_values = torch.bmm(att_weight, values)

        x = self.layernorm1(x + self.dropout1(att_values)) # add & Norm
        FFN_x = self.FFN_linear1(x)
        FFN_x = self.FFN_relu(FFN_x)
        FFN_x = self.FFN_dropout(FFN_x)
        FFN_x = self.FFN_linear2(FFN_x)
        x = self.layernorm2(x + self.dropout2(FFN_x))
        x = x.transpose(1,2)
        return x 


if __name__=='__main__':
    mha = MultiHeadAttention(1,2,2,2)
    modules =  mha.modules()
    print(type(modules))
    print(modules)
    for m in mha.modules():
        print(m)