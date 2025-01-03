#!/usr/bin/env python
# coding: utf-8

# # Scaled Dot-Product Attention

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None, dropout=None):
    """
    Scaled Dot-Product Attention 구현
    Args:
        Q (Tensor): Query, shape [batch_size, seq_len, d_k]
        K (Tensor): Key, shape [batch_size, seq_len, d_k]
        V (Tensor): Value, shape [batch_size, seq_len, d_v]
        mask (Tensor, optional): 마스크 텐서, shape [batch_size, seq_len, seq_len]

    Returns:
        output (Tensor): Attention 결과, shape [batch_size, seq_len, d_v]
        attention_weights (Tensor): Attention 가중치, shape [batch_size, seq_len, seq_len]
    """
    
    # Query와 Key의 행렬곱
    d_k = Q.size(-1)
    attention_scores = torch.matmul(Q, K.transpose(-2, -1))  # 마지막 2개 차원을 전치
    attention_scores = attention_scores / (d_k**0.5)  # Scaling 적용

    # 마스크 적용
    if mask is not None:
        attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

    # Attention weights 계산 (Softmax)
    attention_weights = F.softmax(attention_scores, dim=-1) # [batch_size, seq_len, seq_len]
    if dropout is not None:
        attention_weights = dropout(attention_weights)

    # Attention weights와 Value의 가중합 계산
    output = torch.matmul(attention_weights, V) # [batch_size, seq_len, d_v]
    return output, attention_weights


# # Multi-Head Attention

# In[2]:


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        """
        Multi-Head Attention 초기화
        Args:
            d_model (int): 전체 모델 차원
            n_head (int): Head의 개수
            dropout (float): 드롭아웃 비율
        """
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head  # 각 Head의 차원

        # Query, Key, Value에 대한 선형 변환 계층
        self.Q = nn.Linear(d_model, d_model, bias=False)
        self.K = nn.Linear(d_model, d_model, bias=False)
        self.V = nn.Linear(d_model, d_model, bias=False)
        self.Output = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(p=dropout)

        # Attention 가중치를 저장할 변수
        self.attention = None

    def forward(self, Q, K, V, mask=None):
        """
        Multi-Head Attention Forward 연산
        Args:
            Q (Tensor): Query, shape [batch_size, seq_len, d_model]
            K (Tensor): Key, shape [batch_size, seq_len, d_model]
            V (Tensor): Value, shape [batch_size, seq_len, d_model]
            mask (Tensor, optional): 마스크 텐서, shape [batch_size, seq_len, seq_len]

        Returns:
            output (Tensor): Multi-Head Attention 결과, shape [batch_size, seq_len, d_model]
            attention_weights (Tensor): Attention 가중치, shape [batch_size, n_head, seq_len, seq_len]
        """
        batch_size = Q.size(0)        
        
        # Query, Key, Value를 Head 개수(n_head)로 나누고 차원을 변환
        Q = self.Q(Q).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        K = self.K(K).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        V = self.V(V).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)

        # 마스크 차원을 Head에 맞게 확장
        if mask is not None:
            mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]

        # Scaled Dot-Product Attention 적용
        x, self.attention = scaled_dot_product_attention(Q, K, V, mask=mask, dropout=self.dropout)

        # Head를 결합(concatenate)하고 최종 선형 변환 적용
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.Output(x)

        return output, self.attention


# # Token Embedding

# In[3]:


import torch
import torch.nn as nn
import math

class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model):
        """
        임베딩 계층 초기화
        Args:
            d_model (int): 모델 차원
            vocab_size (int): 어휘 크기, 단어들의 토큰 개수
        """
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.d_model = d_model

    def forward(self, x):
        """
        임베딩 벡터와 스케일링 반환
        Args:
            x (Tensor): 입력 시퀀스, shape [batch_size, seq_len]

        Returns:
            Tensor: 스케일된 임베딩 벡터, shape [batch_size, seq_len, d_model]
        """
        # 임베딩 계산 및 스케일링 적용
        return self.embedding(x) * math.sqrt(self.d_model)


# # Positional Encoding

# In[4]:


import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        Positional Encoding 초기화
        Args:
            d_model (int): 모델 차원
            max_len (int): 최대 시퀀스 길이
        """
        super().__init__()
        
        # Positional Encoding 초기화
        self.encoding = torch.zeros(max_len, d_model)
        
        # 위치 인덱스 생성
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 주기적 함수의 분모 계산
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        
        # 짝수 인덱스: sin, 홀수 인덱스: cos
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)  # 배치 차원 추가

    def forward(self, x):
        """
        Positional Encoding을 입력에 추가
        Args:
            x (Tensor): 입력 텐서, shape [batch_size, seq_len, d_model]
        Returns:
            Tensor: Positional Encoding이 추가된 텐서
        """
        # 디바이스 일치화
        self.encoding = self.encoding.to(x.device)
        
        # Positional Encoding 추가
        x = x + self.encoding[:, :x.size(1), :]
        return x


# # Position-wise Feed-Forward

# In[5]:


import torch
import torch.nn as nn

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed Forward Network
    Args:
        d_model (int): 모델 차원
        d_ff (int): Feed Forward 레이어 내부 차원
        dropout (float): 드롭아웃 비율
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_ff)  # 첫 번째 선형 변환
        self.W2 = nn.Linear(d_ff, d_model)  # 두 번째 선형 변환
        self.dropout = nn.Dropout(dropout)  # 드롭아웃 레이어
        self.relu = nn.ReLU()  # 활성화 함수

    def forward(self, X):
        """
        Args:
            X (Tensor): 입력 텐서, shape [batch_size, seq_len, d_model]
        Returns:
            Tensor: 출력 텐서, shape [batch_size, seq_len, d_model]
        """
        # 첫 번째 선형 변환과 활성화 함수
        X = self.W1(X)
        X = self.relu(X)

        # 드롭아웃 적용
        X = self.dropout(X)

        # 두 번째 선형 변환
        X = self.W2(X)
        return X


# # Add & Norm

# In[6]:


# Add & Norm

import torch
import torch.nn as nn

class AddAndNorm(nn.Module):
    """
    Residual connection + Layer Normalization + Dropout
    """
    def __init__(self, d_model, dropout=0.1):
        """
        Args:
            d_model (int): 모델 차원
            dropout (float): Dropout 비율
        """
        super(AddAndNorm, self).__init__()
        self.norm = nn.LayerNorm(d_model)  # Layer Normalization
        self.dropout = nn.Dropout(dropout)  # Dropout

    def forward(self, x, sublayer=None):
        """
        Args:
            x (Tensor): Residual 입력 텐서, shape [batch_size, seq_len, d_model]
            sublayer (Callable, optional): 적용할 서브 레이어 (없을 경우 None)
        Returns:
            Tensor: Residual Connection + Dropout + Normalization 결과
        """
        if sublayer is None:
            # Sublayer가 없으면 Normalization만 적용
            return self.norm(x)
        
        # Sublayer를 적용하기 전에 입력 x를 정규화
        sublayer_output = self.norm(sublayer)        
        
        # Dropout과 Residual Connection 적용
        return x + self.dropout(sublayer_output)


# # EncoderLayer

# In[7]:


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout=0.1):
        """
        Args:
            d_model (int): 모델 차원
            n_head (int): Multi-Head Attention의 헤드 개수
            d_ff (int): Feed Forward Network의 내부 차원
            dropout (float): Dropout 비율
        """
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_head, dropout)
        self.add_and_norm_1 = AddAndNorm(d_model, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.add_and_norm_2 = AddAndNorm(d_model, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x (Tensor): 입력 텐서, shape [batch_size, seq_len, d_model]
            mask (Tensor): Attention 마스크 텐서, shape [batch_size, seq_len, seq_len]
        Returns:
            Tensor: Encoder Layer의 출력, shape [batch_size, seq_len, d_model]
        """
        # Self-Attention + Add & Norm
        attention_output, _ = self.self_attention(x, x, x, mask)
        x = self.add_and_norm_1(x, attention_output)
        
        # Feed Forward + Add & Norm
        ff_output = self.feed_forward(x)
        x = self.add_and_norm_2(x, ff_output)
        return x



# # Encoder

# In[8]:


import torch
import torch.nn as nn
import math

class Encoder(nn.Module):
    def __init__(self, d_model, n_head, d_ff, num_layers, vocab_size, max_len=5000, dropout=0.1):
        """
        Args:
            d_model (int): 모델 차원
            n_head (int): Multi-Head Attention의 헤드 개수
            d_ff (int): Feed Forward Network의 내부 차원
            num_layers (int): Encoder Layer의 개수
            vocab_size (int): 어휘 크기
            max_len (int): 입력 시퀀스의 최대 길이
            dropout (float): Dropout 비율
        """
        super().__init__()
        
        # 임베딩 레이어
        self.embedding = Embeddings(vocab_size, d_model)
        
        # Positional Encoding
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        
        # 다중 Encoder Layer를 포함하는 리스트
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_head, d_ff, dropout)  # d_k, d_v 제거
            for _ in range(num_layers)
        ])       
        
        # 드롭아웃
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x (Tensor): 입력 시퀀스 텐서, shape [batch_size, seq_len]
            mask (Tensor): Attention 마스크 텐서, shape [batch_size, seq_len, seq_len]
        
        Returns:
            Tensor: 정규화된 Encoder 출력, shape [batch_size, seq_len, d_model]
        """
        # 임베딩 + Positional Encoding
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # 각 Encoder Layer를 순차적으로 통과
        for layer in self.layers:
            x = layer(x, mask)
        return x     


# # Decoder Layer

# In[9]:


class DecoderLayer(nn.Module):
    """
    Transformer Decoder Layer
    Self-Attention, Encoder-Decoder Attention, Feed Forward로 구성
    """
    def __init__(self, d_model, n_head, d_ff, dropout=0.1):
        """
        Args:
            d_model (int): 모델 차원
            n_head (int): Multi-Head Attention의 헤드 개수
            d_ff (int): Feed Forward Network의 내부 차원
            dropout (float): Dropout 비율
        """
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_head, dropout)
        self.add_and_norm_1 = AddAndNorm(d_model, dropout)
        self.encoder_decoder_attention = MultiHeadAttention(d_model, n_head, dropout)
        self.add_and_norm_2 = AddAndNorm(d_model, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.add_and_norm_3 = AddAndNorm(d_model, dropout)

    def forward(self, x, encoder_output, tgt_mask=None, memory_mask=None):
        """
        Args:
            x (Tensor): Decoder 입력 텐서, shape [batch_size, tgt_seq_len, d_model]
            encoder_output (Tensor): Encoder 출력 텐서, shape [batch_size, src_seq_len, d_model]
            tgt_mask (Tensor, optional): Decoder의 Self-Attention 마스크
            memory_mask (Tensor, optional): Encoder-Decoder Attention 마스크
        
        Returns:
            Tensor: Decoder Layer 출력, shape [batch_size, tgt_seq_len, d_model]
        """
        # Self-Attention + Add & Norm
        self_attention_output, _ = self.self_attention(x, x, x, tgt_mask)
        x = self.add_and_norm_1(x, self_attention_output)
        
        # Encoder-Decoder Attention + Add & Norm
        eda_output, _ = self.encoder_decoder_attention(x, encoder_output, encoder_output, memory_mask)
        x = self.add_and_norm_2(x, eda_output)
        
        # Feed Forward + Add & Norm
        ff_output = self.feed_forward(x)
        x = self.add_and_norm_3(x, ff_output)
        
        return x


# # Decoder

# In[10]:


class Decoder(nn.Module):
    """
    Transformer Decoder
    """
    def __init__(self, d_model, n_head, d_ff, num_layers, vocab_size, max_len=5000, dropout=0.1):
        """
        Args:
            d_model (int): 모델 차원
            n_head (int): Multi-Head Attention의 헤드 개수
            d_ff (int): Feed Forward Network의 내부 차원
            num_layers (int): Decoder Layer의 개수
            vocab_size (int): 어휘 크기
            max_len (int): 입력 시퀀스의 최대 길이
            dropout (float): Dropout 비율
        """
        super().__init__()
        self.embedding = Embeddings(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([ DecoderLayer(d_model, n_head, d_ff, dropout) for _ in range(num_layers) ])
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, encoder_output, tgt_mask=None, memory_mask=None):
        """
        Args:
            tgt (Tensor): Decoder 입력 텐서, shape [batch_size, tgt_seq_len]
            encoder_output (Tensor): Encoder 출력 텐서, shape [batch_size, src_seq_len, d_model]
            tgt_mask (Tensor, optional): Decoder의 Self-Attention 마스크
            memory_mask (Tensor, optional): Encoder-Decoder Attention 마스크

        Returns:
            Tensor: 정규화된 Decoder 출력, shape [batch_size, tgt_seq_len, d_model]
        """
        x = self.embedding(tgt)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, encoder_output, tgt_mask, memory_mask)
        
        x = self.layer_norm(x)
        return x


# # Transformer 만들기 

# In[11]:


import torch
import torch.nn as nn
import math


class MyTransformer(nn.Module):
    """
    Complete Transformer Model
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, n_head, d_ff, num_encoder_layers, num_decoder_layers, max_len=5000, dropout=0.1):
        """
        Args:
            src_vocab_size (int): 소스 어휘 크기
            tgt_vocab_size (int): 타겟 어휘 크기
            d_model (int): 모델 차원
            n_head (int): Multi-Head Attention의 헤드 개수
            d_ff (int): Feed Forward Network의 내부 차원
            num_encoder_layers (int): Encoder Layer의 개수
            num_decoder_layers (int): Decoder Layer의 개수
            max_len (int): 입력 시퀀스의 최대 길이
            dropout (float): Dropout 비율
        """
        super().__init__()
        self.encoder = Encoder(d_model, n_head, d_ff, num_encoder_layers, src_vocab_size, max_len, dropout)
        self.decoder = Decoder(d_model, n_head, d_ff, num_decoder_layers, tgt_vocab_size, max_len, dropout)
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)  # 최종 출력층
        self.softmax = nn.Softmax(dim=-1)  # Softmax

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None, inference=False):
        """
        Args:
            src (Tensor): 소스 입력 텐서, shape [batch_size, src_seq_len]
            tgt (Tensor): 타겟 입력 텐서, shape [batch_size, tgt_seq_len]
            src_mask (Tensor, optional): Encoder Self-Attention 마스크
            tgt_mask (Tensor, optional): Decoder Self-Attention 마스크
            memory_mask (Tensor, optional): Encoder-Decoder Attention 마스크

        Returns:
            Tensor: Transformer 출력, shape [batch_size, tgt_seq_len, tgt_vocab_size]
        """
        # Encoder: 소스 입력 처리
        encoder_output = self.encoder(src, src_mask)

        # Decoder: 타겟 입력 처리
        decoder_output = self.decoder(tgt, encoder_output, tgt_mask, memory_mask)

        # Output Layer: 최종 출력 생성
        output = self.output_layer(decoder_output)
        if inference:  # 추론 시 softmax 적용
            return self.softmax(output)
        return output  # 학습 시 raw logits 반환


# # 시험데이터 준비

# In[18]:


import pandas as pd

# 간단한 영어-한국어 toy dataset
data = [
    ("i like apples", "나는 사과를 좋아한다."),
    ("you like apples", "너는 사과를 좋아한다."),
    ("he like apples", "그는 사과를 좋아한다."),
    ("she like apples", "그녀는 사과를 좋아한다."),
    ("we like apples", "우리는 사과를 좋아한다."),
    ("they like apples", "그들은 사과를 좋아한다."),
    ("i hate apples", "나는 사과를 싫어한다."),
    ("you hate apples", "너는 사과를 싫어한다."),
    ("he hates apples", "그는 사과를 싫어한다."),
    ("she hates apples", "그녀는 사과를 싫어한다."),
    ("we hate apples", "우리는 사과를 싫어한다."),
    ("they hate apples", "그들은 사과를 싫어한다."),
    ("i see apples", "나는 사과를 본다."),
    ("you see apples", "너는 사과를 본다."),
    ("he sees apples", "그는 사과를 본다."),
    ("she sees apples", "그녀는 사과를 본다."),
    ("we see apples", "우리는 사과를 본다."),
    ("they see apples", "그들은 사과를 본다."),
    ("i see a dog", "나는 개를 본다."),
    ("you see a dog", "너는 개를 본다."),
    ("he sees a dog", "그는 개를 본다."),
    ("she sees a dog", "그녀는 개를 본다."),
    ("we see a dog", "우리는 개를 본다."),
    ("they see a dog", "그들은 개를 본다."),
    ("i like books", "나는 책을 좋아한다."),
    ("you like books", "너는 책을 좋아한다."),
    ("he likes books", "그는 책을 좋아한다."),
    ("she likes books", "그녀는 책을 좋아한다."),
    ("we like books", "우리는 책을 좋아한다."),
    ("they like books", "그들은 책을 좋아한다."),
    ("i want a book", "나는 책을 원한다."),
    ("you want a book", "너는 책을 원한다."),
    ("he wants a book", "그는 책을 원한다."),
    ("she wants a book", "그녀는 책을 원한다."),
    ("we want a book", "우리는 책을 원한다."),
    ("they want a book", "그들은 책을 원한다."),
    ("i hate a dog", "나는 개를 싫어한다."),
    ("you hate a dog", "너는 개를 싫어한다."),
    ("he hates a dog", "그는 개를 싫어한다."),
    ("she hates a dog", "그녀는 개를 싫어한다."),
    ("we hate a dog", "우리는 개를 싫어한다."),    
    ("they hate a dog", "그들은 개를 싫어한다."),
    ("i read a book", "나는 책을 읽는다."),
    ("you read a book", "너는 책을 읽는다."),
    ("he reads a book", "그는 책을 읽는다."),
    ("she reads a book", "그녀는 책을 읽는다."),
    ("we read a book", "우리는 책을 읽는다."),
    ("they read a book", "그들은 책을 읽는다."),
    ("i like a cat", "나는 고양이를 좋아한다."),
    ("you like a cat", "너는 고양이를 좋아한다."),
    ("he like a cat", "그는 고양이를 좋아한다."),
    ("she like a cat", "그녀는 고양이를 좋아한다."),
    ("we like a cat", "우리는 고양이를 좋아한다."),
    ("they like a cat", "그들은 고양이를 좋아한다."),
    ("i hate a cat", "나는 고양이를 싫어한다."),
    ("you hate a cat", "너는 고양이를 싫어한다."),
    ("he hates a cat", "그는 고양이를 싫어한다."),
    ("she hates a cat", "그녀는 고양이를 싫어한다."),
    ("we hate a cat", "우리는 고양이를 싫어한다."),
    ("they hate a cat", "그들은 고양이를 싫어한다."),
    ("i want a dog", "나는 개를 원한다."),
    ("you want a dog", "너는 개를 원한다."),
    ("he want a dog", "그는 개를 원한다."),
    ("she want a dog", "그녀는 개를 원한다."),
    ("we want a dog", "우리는 개를 원한다."),
    ("they want a dog", "그들은 개를 원한다."),
    ("i want a cat", "나는 고양이를 원한다."),
    ("you want a cat", "너는 고양이를 원한다."),
    ("he want a cat", "그는 고양이를 원한다."),
    ("she want a cat", "그녀는 고양이를 원한다."),
    ("we want a cat", "우리는 고양이를 원한다."),
    ("they want a cat", "그들은 고양이를 원한다."),
    ("i see a cat", "나는 고양이를 본다."),
    ("you see a cat", "너는 고양이를 본다."),
    ("he sees a cat", "그는 고양이를 본다."),
    ("she sees a cat", "그녀는 고양이를 본다."),
    ("we see a cat", "우리는 고양이를 본다."),
    ("they see a cat", "그들은 고양이를 본다."),
]



# 단어 사전 생성
def build_vocab(sentences):
    # pad : 문자의 길이, sos : 문장의 시작, eos 문장의 끝을 나타내는 토큰
    vocab = {"<pad>": 0, "<sos>": 1, "<eos>": 2}
    for sentence in sentences:
        for word in sentence.split():
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab

# 영어와 프랑스어 단어 사전
src_vocab = build_vocab([src for src, tgt in data])
tgt_vocab = build_vocab([tgt for src, tgt in data])

src_vocab_table = pd.DataFrame( { "Index": list(src_vocab.values()),"Token": list(src_vocab.keys()) })
display(src_vocab_table.set_index('Index').T)

tgt_vocab_table = pd.DataFrame( { "Index": list(tgt_vocab.values()),"Token": list(tgt_vocab.keys()) })
display(tgt_vocab_table.set_index('Index').T)


# 데이터 전처리: 텍스트 -> 정수 인덱스 변환
def preprocess_data(data, src_vocab, tgt_vocab):
    src_data, tgt_data = [], []
    for src, tgt in data:
        src_indices = [src_vocab[word] for word in src.split()] + [src_vocab["<eos>"]]
        tgt_indices = [tgt_vocab["<sos>"]] + [tgt_vocab[word] for word in tgt.split()] + [tgt_vocab["<eos>"]]
        src_data.append(src_indices)
        tgt_data.append(tgt_indices)
    return src_data, tgt_data

src_data, tgt_data = preprocess_data(data, src_vocab, tgt_vocab)

# 패딩 함수
def pad_sequences(sequences, pad_idx):
    max_len = max(len(seq) for seq in sequences)
    return [seq + [pad_idx] * (max_len - len(seq)) for seq in sequences]

src_data = pad_sequences(src_data, src_vocab["<pad>"])
tgt_data = pad_sequences(tgt_data, tgt_vocab["<pad>"])

# 텐서 변환
src_tensor = torch.tensor(src_data)
tgt_tensor = torch.tensor(tgt_data)


print("Source Tensor:", len(src_tensor))
print("Target Tensor:", len(tgt_tensor))


# # 훈련 루프 작성

# In[19]:


# DataLoader를 활용한 배치 처리
from torch.utils.data import DataLoader, TensorDataset

# 하이퍼파라미터 설정
src_vocab_size = len(src_vocab)
tgt_vocab_size = len(tgt_vocab)
max_len = 20
batch_size = 4

d_model = 128  # 모델의 차원 수
n_head = 8     # Multi-Head Attention 헤드 수
d_ff = 256     # Feed Forward 네트워크의 은닉층 차원 수
num_encoder_layers = 6 # Encoder Nx 반복횟수
num_decoder_layers = 6 # Decoder Nx 반복횟수
learning_rate = 0.001   
num_epochs = 100

# 데이터셋 생성 및 DataLoader 정의
dataset = TensorDataset(src_tensor, tgt_tensor)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Padding Mask 생성
def generate_padding_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)  # Shape: [batch_size, 1, 1, seq_len]


# 모델 초기화
model = MyTransformer(src_vocab_size, tgt_vocab_size, d_model, n_head, d_ff, num_encoder_layers, num_decoder_layers, max_len)
criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab["<pad>"])
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 학습 루프
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for batch in data_loader:
        src, tgt = batch
        tgt_input = tgt[:, :-1]  # <eos> 제외
        tgt_labels = tgt[:, 1:]  # <sos> 제외

        # 모델 출력
        output = model(src, tgt_input)

        # Loss 계산
        output = output.reshape(-1, tgt_vocab_size)  # view 대신 reshape로 일관성 유지
        tgt_labels = tgt_labels.reshape(-1)
        loss = criterion(output, tgt_labels)

        # 경사하강
        optimizer.zero_grad()
        
        # 역전파 
        loss.backward()
        
        # 파라미터 최적호
        optimizer.step()

        epoch_loss += loss.item()

    # 10 에포크마다 Loss 출력
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(data_loader):.4f}")


# # TEST

# In[20]:


import pandas as pd
tgt_vocab_table = pd.DataFrame( { "Index": list(tgt_vocab.values()),"Token": list(tgt_vocab.keys()) })
display(tgt_vocab_table.set_index('Index').T)

def translate(model, sentence, src_vocab, tgt_vocab):
    model.eval()

    # 입력 전처리
    src_indices = [src_vocab[word] for word in sentence.split()] + [src_vocab["<eos>"]]
    src_tensor = torch.tensor([src_indices])

    # 초기 타겟 시퀀스 (<sos> 토큰 포함)
    tgt_indices = [tgt_vocab["<sos>"]]
    tgt_tensor = torch.tensor([tgt_indices])

    for _ in range(max_len):
        # 모델 예측
        output = model(src_tensor, tgt_tensor, inference=True)
        probs = output[:, -1, :]  # 마지막 타임스텝
        next_token = probs.argmax(dim=-1).item()
        tgt_indices.append(next_token)
        # <eos> 토큰 생성 시 종료
        if next_token == tgt_vocab["<eos>"]:            
            break

        # 생성된 토큰 추가
        
        tgt_tensor = torch.tensor([tgt_indices])

    # 정수 인덱스를 단어로 변환
    tgt_vocab_inv = {v: k for k, v in tgt_vocab.items()}
    translation = " ".join([tgt_vocab_inv[idx] for idx in tgt_indices[1:-1]])  # <sos> 제외
    print(tgt_indices, [tgt_vocab_inv[idx] for idx in tgt_indices])
    
    print(f"Input: {sentence} : Translation: {translation}")    
    print()    
    return translation

# 번역 예제
sentence = "i like a dog"
translation = translate(model, sentence, src_vocab, tgt_vocab)
sentence = "they want apples"
translation = translate(model, sentence, src_vocab, tgt_vocab)
sentence = "you see a book"
translation = translate(model, sentence, src_vocab, tgt_vocab)


# In[ ]:




