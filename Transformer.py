import math
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        # 计算逆频率
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer('inv_freq', inv_freq)

        # 缓存最大序列长度
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)

        # 计算频率
        freqs = torch.einsum('i, j->ij', t, self.inv_freq)

        # 计算嵌入
        emb = torch.cat((freqs, freqs), dim=-1)
        dtype = torch.get_default_dtype()
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        # 如果序列长度大于缓存的最大序列长度，则重新计算
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum('i , j->ij', t, self.inv_freq)

            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer('cos_cached', emb.cos()[None, None, :, :], persistent=False)
            self.register_buffer('sin_cached', emb.sin()[None, None, :, :], persistent=False)

        # 返回缓存的余弦和正弦值
        return (self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype), self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype))

def rotate_half(x):
    # 将输入张量分为两部分，并交换顺序
    x1 = x[..., :x.shape[-1]//2]
    x2 = x[..., x.shape[-1]//2:]

    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # 将余弦和正弦值从缓存中取出
    cos = cos.squeeze(1).squeeze(0)
    sin = sin.squeeze(1).squeeze(0)
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    # 计算查询和键的嵌入
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048, dropout=0.1):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=0.1)

        position = torch.arange(0, max_len)
        div_term = 1.0 / torch.pow(10000, torch.arange(0, d_model, 2) / d_model)
        pe = torch.zeros(max_len, d_model)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(1)    # [max_len, bs, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x):

        pe = self.pe[:x.size(0), ...]
        x = x + pe

        return x

def get_attn_pad_mask(seq_q, seq_k):

    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()

    # 创建一个全为1的张量
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)

    # [batch_size, tgt_len, tgt_len] 或 [batch_size, src_len, src_len] 或 [batch_size, tgt_len, src_len]
    return pad_attn_mask.expand(batch_size, len_q, len_k)

def get_attn_subsequence_mask(seq):
    """
    :param seq: [batch_size, tgt_len]
    :return: 序列掩码的位置
    """
    """
    防止decoder看到未来的信息，在t时刻，解码输出只能依赖于t时刻之前的输出
    """
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]

    subsequence_mask = np.triu(np.ones(attn_shape), k=1)    # 上三角矩阵，不包括对角线，要遮掩的部分
    subsequence_mask = torch.from_numpy(subsequence_mask).bool()

    # [batch_size, tgt_len, tgt_len]
    return subsequence_mask

# 两个大组件，多头注意力机制以及前馈神经网络
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        """
        :param Q: [batch_size, n_heads, len_q, d_q]
        :param K: [batch_size, n_heads, len_k, d_k]
        :param V: [batch_size, n_heads, len_v, d_v]
        :param attn_mask: [batch_size, n_heads, len_q, len_k]
        :return:
        """
        d_k = K.size(-1)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # [batch_size, n_heads, len_q, len_k]
        """
        这里也解释了为什么get_attn_pad_mask中是[batch_size, tgt_len, src_len]
        """
        scores.masked_fill_(attn_mask, -1e9)

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)

        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()

        self.n_heads = n_heads
        self.d_model = d_model
        self.d_q = self.d_k = self.d_v = int(d_model / n_heads)
        self.W_Q = nn.Linear(d_model, self.d_q * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, self.d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, self.d_v * n_heads, bias=False)
        # 全连接，合并
        self.fc = nn.Linear(n_heads * self.d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        """
        :param input_Q: [batch_size, len_q, d_model]
        :param input_K: [batch_size, len_k, d_model]
        :param input_V: [batch_size, len_v, d_model]
        :param attn_mask: [batch_size, seq_len, seq_len]
        :return:
        这些参数，在Transformer的三个多头注意力机制中各不相同
        """
        residual = input_Q
        batch_size = input_Q.size(0)

        # [batch_size, n_heads, len_q, d_q]
        Q = self.W_Q(input_Q).view(batch_size, self.n_heads, -1, self.d_q)
        K = self.W_K(input_K).view(batch_size, self.n_heads, -1, self.d_k)
        V = self.W_V(input_V).view(batch_size, self.n_heads, -1, self.d_v)

        # [batch_size, n_heads, seq_len, seq_len]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)

        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)

        # [batch_size, n_heads, len_q, d_v] -> [batch_size, len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v)

        # [batch_size, len_q, d_model]
        output = self.fc(context)

        return nn.LayerNorm(self.d_model).to(device)(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()

        self.d_model = d_model
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )

    def forward(self, inputs):
        """
        :param inputs: MultiHeadAttention层的输出
        :return:
        """
        residual = inputs

        # [batch_size, len_q, d_model]
        outputs = self.fc(inputs)

        return nn.LayerNorm(self.d_model)(outputs + residual)

# Transformer架构
# Encoder_Layer -> Encoder
# Decoder_Layer -> Decoder
# Encoder + Decoder -> Transformer

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, enc_inputs, enc_self_attn_mask):
        """
        :param enc_inputs: [batch_size, src_len, d_model]
        :param enc_self_attn_mask: [batch_size, src_len, src_len]
        :return: [batch_size, src_len, d_model]
        """
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)

        enc_outputs = self.pos_ffn(enc_outputs)

        return enc_outputs, attn

class Encoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, n_layers, src_vocab_size):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)])

    def forward(self, enc_inputs):

        enc_outputs = self.src_emb(enc_inputs)
        # [src_len, batch_size, d_model] -> [batch_size, src_len, d_model]
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)

        # Encoder中的pad_mask
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
        enc_self_attns = []     # 保存attention的值，方便画热力图

        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)

        return enc_outputs, enc_self_attns


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(d_model, n_heads)
        self.enc_dec_attn = MultiHeadAttention(d_model, n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)
        """
        decoder中多了一个enc_dec交互的多头注意力机制
        """

    def forward(self, enc_outputs, dec_inputs, dec_self_attn_mask, enc_dec_attn_mask):
        """
        :param enc_outputs: [batch_size, src_len, d_model]
        :param dec_inputs: [batch_size, tgt_len, d_model]
        :param dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        :param enc_dec_attn_mask: [batch_size, tgt_len, src_len]
        :return:
        """

        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)

        dec_outputs, dec_enc_attn = self.enc_dec_attn(dec_outputs, enc_outputs, enc_outputs, enc_dec_attn_mask)

        dec_outputs = self.pos_ffn(dec_outputs)

        return dec_outputs, dec_self_attn, dec_enc_attn


class Decoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, n_layers, tgt_vocab_size):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):

        dec_outputs = self.tgt_emb(dec_inputs)

        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1)

        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs)

        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask), 0)

        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)

        dec_self_attns, dec_enc_attns = [], []

        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(enc_outputs, dec_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)

        return dec_outputs, dec_self_attns, dec_enc_attns

class Transformer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, n_layers, src_vocab_size, tgt_vocab_size):
        super(Transformer, self).__init__()
        self.encoder = Encoder(d_model, n_heads, d_ff, n_layers, src_vocab_size)
        self.decoder = Decoder(d_model, n_heads, d_ff, n_layers, tgt_vocab_size)
        self.fc = nn.Linear(d_model, tgt_vocab_size, bias=False)

    def forward(self, enc_inputs, dec_inputs):

        enc_outputs, enc_self_attns = self.encoder(enc_inputs)

        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)

        logits = self.fc(dec_outputs)

        return nn.Softmax(-1)(logits.view(-1, logits.size(-1))), enc_self_attns, dec_self_attns, dec_enc_attns


def make_data(sentences):
    """把单词序列转换为数字序列"""
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    for i in range(len(sentences)):
        enc_input = [[src_vocab[n] for n in sentences[i][0].split()]]
        dec_input = [[tgt_vocab[n] for n in sentences[i][1].split()]]
        dec_output = [[tgt_vocab[n] for n in sentences[i][2].split()]]

        # [[1, 2, 3, 4, 5, 6, 7, 0], [1, 2, 8, 4, 9, 6, 7, 0], [1, 2, 3, 4, 10, 6, 7, 0]]
        enc_inputs.extend(enc_input)
        # [[9, 1, 2, 3, 4, 5, 11], [9, 1, 2, 6, 7, 5, 11], [9, 1, 2, 3, 8, 5, 11]]
        dec_inputs.extend(dec_input)
        # [[1, 2, 3, 4, 5, 11, 10], [1, 2, 6, 7, 5, 11, 10], [1, 2, 3, 8, 5, 11, 10]]
        dec_outputs.extend(dec_output)

    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)

class MyDataSet(Dataset):
    """自定义DataLoader"""

    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]



if __name__ == '__main__':

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    sentences = [
        # 中文和英语的单词个数不要求相同
        # enc_input                dec_input           dec_output
        ['我 有 一 个 好 朋 友 P', 'S I have a good friend .', 'I have a good friend . E'],
        ['我 有 零 个 女 朋 友 P', 'S I have zero girl friend .', 'I have zero girl friend . E'],
        ['我 有 一 个 男 朋 友 P', 'S I have a boy friend .', 'I have a boy friend . E']
    ]

    # 测试集（希望transformer能达到的效果）
    # 输入："我 有 一 个 女 朋 友"
    # 输出："i have a girlfriend"

    # 中文和英语的单词要分开建立词库
    # Padding Should be Zero
    src_vocab = {'P': 0, '我': 1, '有': 2, '一': 3,
                 '个': 4, '好': 5, '朋': 6, '友': 7, '零': 8, '女': 9, '男': 10}
    src_idx2word = {i: w for i, w in enumerate(src_vocab)}
    src_vocab_size = len(src_vocab)

    tgt_vocab = {'P': 0, 'I': 1, 'have': 2, 'a': 3, 'good': 4,
                 'friend': 5, 'zero': 6, 'girl': 7, 'boy': 8, 'S': 9, 'E': 10, '.': 11}
    idx2word = {i: w for i, w in enumerate(tgt_vocab)}
    tgt_vocab_size = len(tgt_vocab)
    src_len = 8  # （源句子的长度）enc_input max sequence length
    tgt_len = 7  # dec_input(=dec_output) max sequence length

    enc_inputs, dec_inputs, dec_outputs = make_data(sentences)

    loader = DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), batch_size=2, shuffle=True)

    d_model = 512
    n_heads = 8
    d_ff = 2048
    n_layers = 6

    model = Transformer(d_model, n_heads, d_ff, n_layers, src_vocab_size, tgt_vocab_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)

    epochs = 20

    for epoch in range(epochs):
        for enc_inputs, dec_inputs, dec_outputs in loader:

            optimizer.zero_grad()

            outputs, _, _, _ = model(enc_inputs, dec_inputs)

            loss = criterion(outputs, dec_outputs.view(-1))

            print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), "Transformer.pth")



















