import copy
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def clones(module, N):
    """
        decide the layer of decoder and encoder
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    """
        搭建了一个批归一化模型，降低算力要求
    """

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        """
            对输入x进行去中心化，同时a2，b2两个参数可随模型训练调整
        """
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
        A residual connection followed by a layer norm.
        Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
            Apply residual connection to any sublayer with the same size.
        """
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    """
        Encoder is made up of self-attn and feed forward (defined below)
    """

    def __init__(self, size, self_attention, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x_: self.self_attention(x_, x_, x_, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    """
        Generic N layer decoder with masking.
    """

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
            这个mask就像是我们小时候的默写，你得先默写出来再去看答案，如果默写错了，就要吃一竹竿长记性（loss反向传递，系数更新）
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """
        Decoder is made of self-attn, src-attn, and feed forward
    """

    def __init__(self, size, self_attention, src_attention, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attention = self_attention
        self.src_attention = src_attention
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.sublayer[0](x, lambda x_: self.self_attention(x_, x_, x_, tgt_mask))
        x = self.sublayer[1](x, lambda x_: self.src_attention(x_, memory, memory, src_mask))
        return self.sublayer[2](x, self.feed_forward)


def attention(query, key, value, mask=None, dropout=None):
    """
    :param query:
    :param key:
    :param value: vocab vector
    :return: value_(i.e z) and A
    :param dropout:
    :param mask:
    """
    d_k = key.size(-1)  # d_k表示key矩阵的维度
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # key是一个二维矩阵
    # 进行掩码操作
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)  # 按列进行softmax，即计算每个词在语句中出现的概率

    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        """
        :param h: the number of head
        :param d_model: the dimension of the model
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h  # We assume d_v always equals d_k
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attention = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask):
        if mask is not None:
            mask = mask.unsqueeze(1)  # Same mask applied to all h heads.
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [linears(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for linears, x in
                             zip(self.linears, (query, key, value))]  # (30,8,10,64)

        # the shape of query is [nbatches,the num of head,1,dimension of max]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attention = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.d_k * self.h)

        # Linear change on the final result
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model, padding_idx=1)
        self.d_model = d_model

    def forward(self, x):
        # Since dividing sqrt(d_model) in self-attention, transformer need to multiply it back in the decoding phase
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
        max_len decide the size of word matrix
    """

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)  # 矩阵转置
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-(math.log(10000.0) / d_model)))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)  # Set the location code unchanged

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class EncoderDecoder(nn.Module):
    """
       A standard Encoder-Decoder architecture. (Base for this and many other models.)
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed  # 源语言嵌入层
        self.tgt_embed = tgt_embed  # 目标语言嵌入层
        # 俩个语言嵌入层必定通过算法融合得到一个新的”潜空间“
        self.generator = generator

    def forward(self, src, src_mask, tgt, tgt_mask):
        """
            Take in and process masked src and target sequences.
            前向传播训练过程
            src: (30,10)
            tgt: (30,9)
            src_mask: (30,1,10)
            tgt_mask: (30,9,9)
        """
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        """
            把模型训练得到的训练维度还原为词向量原始维度
        """
        super(Generator, self).__init__()
        self.linear = nn.Linear(d_model, vocab)

    def forward(self, x):

        return F.log_softmax(self.linear(x), dim=-1)


def transformer(src_vocab_num, tgt_vocab_num, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy
    multi_head_attention = MultiHeadAttention(h, d_model)
    FFD = PositionwiseFeedForward(d_model, d_ff)
    PE = PositionalEncoding(d_model, dropout)

    model = EncoderDecoder(
        Encoder(
            EncoderLayer(size=d_model, self_attention=c(multi_head_attention), feed_forward=c(FFD), dropout=dropout),
            N),
        Decoder(
            DecoderLayer(size=d_model, self_attention=c(multi_head_attention), src_attention=c(multi_head_attention),
                         feed_forward=FFD, dropout=dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab_num), c(PE)),
        nn.Sequential(Embeddings(d_model, tgt_vocab_num), c(PE)),
        Generator(d_model, tgt_vocab_num))

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        # 过滤偏差层
        if p.dim() > 1:
            nn.init.xavier_normal_(p)

    return model
