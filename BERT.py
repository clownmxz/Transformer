import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 第一块：词嵌入，BERT相比Transformer而言多了一个分块嵌入(segment embedding)
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = 1.0 / torch.pow(10000, torch.arange(0, d_model, 2, dtype=torch.float32) / d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # pe: [max_len, d_model] -> [max_len, batch_size, d_model]
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        :param x: 一般应该是，token的embedding，[seq_len, batch_size, d_model]
        :return: [batch_size, seq_len, d_model]
        """
        return self.pe[0:x.size(0), :].transpose(0, 1)

class BERTEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, dropout=0.1):
        super(BERTEmbedding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        self.position = PositionalEmbedding(d_model)
        self.token = nn.Embedding(vocab_size, d_model)
        self.segment = nn.Embedding(3, d_model)

    def forward(self, sequence, segment_label):
        """
        :param sequence: [batch_size, seq_len]
        :param segment_label: [batch_size, seq_len]
        :return:
        """
        x = self.token(sequence) + self.position(sequence) + self.segment(segment_label)

        return self.dropout(x)

# 第二块：Transformer的Encoder块
# pad_mask
# pad_mask -> ScaledDotProductAttention
# ScaledDotProductAttention -> MultiHeadAttention
# MultiHeadAttention + PoswiseFeedforwardNet -> EncoderLayer
# EncoderLayer -> Encoder
def get_atten_pad_mask(seq_q, seq_k):
    """
    :param seq_q: [batch_size, seq_q_len]
    :param seq_k: [batch_size, seq_k_len]
    :return:
    """
    """
    这里好像不需要这么写，毕竟和Transformer不一样，BERT的Encoder中seq_q = seq_k
    """
    batch_size, len_q = seq_q.size(0), seq_q.size(1)
    batch_size, len_k = seq_k.size(0), seq_k.size(1)

    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)

    return pad_attn_mask.expand(batch_size, len_q, len_k)

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        """
        :param Q: [batch_size, n_heads, len_q, d_q]
        :param K: [batch_size, n_heads, len_k, d_k]
        :param V: [batch_size, n_heads, len_v, d_v]
        :param attn_mask: [batch_size, n_heads, len_q, len_k]
        :return: [batch_size, n_heads, len_q, d_v]
        """

        d_k = K.size(-1)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        scores.masked_fill_(attn_mask, -1e9)

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)

        return context

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_q = self.d_k = self.d_v = int(d_model // n_heads)

        self.W_Q = nn.Linear(d_model, n_heads * self.d_q)
        self.W_K = nn.Linear(d_model, n_heads * self.d_k)
        self.W_V = nn.Linear(d_model, n_heads * self.d_v)

        self.fc = nn.Linear(n_heads * self.d_v, d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, inputs_Q, inputs_K, inputs_V, attn_mask):
        """
        :param inputs_Q: [batch_size, len_q, d_model]
        :param inputs_K:
        :param inputs_V:
        :param attn_mask: [batch_size, len_q, len_k]
        :return:
        """

        residual = inputs_Q
        batch_size = inputs_Q.size(0)

        # [batch_size, len_q, d_model] -> [batch_size, len_q, n_heads * d_q] -> [batch_size, n_heads, len_q, d_q]
        Q = self.W_Q(inputs_Q).reshape(batch_size, self.n_heads, -1, self.d_q)
        K = self.W_K(inputs_K).reshape(batch_size, self.n_heads, -1, self.d_k)
        V = self.W_V(inputs_V).reshape(batch_size, self.n_heads, -1, self.d_v)

        # [batch_size, len_q, len_k] -> [batch_size, n_heads, len_q, len_k]
        attn_mask = attn_mask.repeat(1, self.n_heads, 1, 1)

        context = ScaledDotProductAttention()(Q, K, V, attn_mask)
        # [batch_size, n_heads, len_q, d_v] -> [batch_size, len_q, n_heads * d_v]
        context = context.reshape(batch_size, -1, self.n_heads * self.d_v)

        output = self.fc(context)

        output = self.dropout(output)

        return nn.LayerNorm(self.d_model)(output + residual)

class PoswiseFeedforwardNet(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PoswiseFeedforwardNet, self).__init__()

        self.d_model = d_model
        # BERT相比于Transformer，其激活函数有所不同为GELU()，而非RELU()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, inputs):
        """
        :param inputs: [batch_size, seq_len, d_model]
        :return:
        """

        residual = inputs

        outputs = self.fc(inputs)
        outputs = self.dropout(outputs)

        return nn.LayerNorm(self.d_model)(residual + outputs)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.multi_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.pos_ffn = PoswiseFeedforwardNet(d_model, d_ff, dropout)

    def forward(self, inputs, attn_mask):
        """
        :param inputs: [batch_size, len_q, d_model]
        :param attn_mask: [batch_size, len_q, len_k]
        :return:
        """

        outputs = self.multi_attention(inputs, inputs, inputs, attn_mask)
        outputs = self.pos_ffn(outputs)

        return outputs

class Encoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, n_layers, vocab_size, dropout=0.1):
        super(Encoder, self).__init__()
        self.inputs_emb = BERTEmbedding(vocab_size, d_model, dropout)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])

    def forward(self, sequence, segment_label):
        """
        :param sequence: [batch_size, seq_len]
        :param segment_label: [batch_size, seq_len]
        :return: [batch_size, seq_len, d_model]
        """

        attn_mask = get_atten_pad_mask(sequence, sequence)

        outputs = self.inputs_emb(sequence, segment_label)

        for layer in self.layers:
            outputs = layer(outputs, attn_mask)

        return outputs

# 第三块：BERT结构，包含MLM和NSP
class NextSentencePrediction(nn.Module):
    def __init__(self, d_model):
        super(NextSentencePrediction, self).__init__()
        self.linear = nn.Linear(d_model, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):

        y = self.linear(x)

        # [batch_size * seq_len, 2]
        return self.softmax(y.reshape(-1, y.size(-1)))

class MaskedLanguageModel(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(MaskedLanguageModel, self).__init__()
        self.linear = nn.Linear(d_model, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):

        y = self.linear(x)

        # [batch_size * seq_len, vocab_size]
        return self.softmax(y.reshape(-1, y.size(-1)))


class BERT(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, n_layers, vocab_size, dropout=0.1):
        super(BERT, self).__init__()
        self.transformer_encoder = Encoder(d_model, n_heads, d_ff, n_layers, vocab_size, dropout=0.1)
        self.mlm = MaskedLanguageModel(d_model, vocab_size)
        self.nsp = NextSentencePrediction(d_model)

    def forward(self, sequence, segment_label):

        outputs = self.transformer_encoder(sequence, segment_label)

        return self.mlm(outputs), self.nsp(outputs)








