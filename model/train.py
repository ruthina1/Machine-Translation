import os
import math
import re
import unicodedata
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# --- Constants ---
PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2
MAX_LENGTH = 10

# --- Vocabulary Class ---
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.index2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.word2count = {}
        self.n_words = 4

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.word2count[word] = 1
            self.n_words += 1
        else:
            self.word2count[word] += 1

# --- Text Normalization Helper Functions ---
def unicode_to_ascii(s):
    for c in unicodedata.normalize('NFD', s):
        if unicodedata.category(c) != 'Mn':
            yield c

def normalize_string(s):
    s = ''.join(unicode_to_ascii(s.lower().strip()))
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s.strip()

def sentence_to_indices(lang, sentence):
    indices = []
    for word in sentence.split(' '):
        token_id = lang.word2index.get(word, 3)  # 3 = <UNK>
        indices.append(token_id)
    indices.append(EOS_TOKEN)
    return indices

def pad_sequence(indices, max_length):
    if len(indices) < max_length:
        return indices + [PAD_TOKEN] * (max_length - len(indices))
    return indices[:max_length - 1] + [EOS_TOKEN]

def prepare_data_loader(pairs, input_lang, output_lang, batch_size=64):
    input_tensors = []
    target_tensors = []

    for pair in pairs:
        input_idx = pad_sequence(sentence_to_indices(input_lang, pair[0]), MAX_LENGTH)
        raw_target_indices = sentence_to_indices(output_lang, pair[1])
        target_idx = pad_sequence([SOS_TOKEN] + raw_target_indices, MAX_LENGTH)

        input_tensors.append(input_idx)
        target_tensors.append(target_idx)

    dataset = TensorDataset(
        torch.tensor(input_tensors, dtype=torch.long),
        torch.tensor(target_tensors, dtype=torch.long)
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return loader

# --- Transformer Building Blocks ---
class PE(nn.Module):
    def __init__(self, d_model, max_len=50):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MHA(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        B, S, _ = q.shape
        _, S_k, _ = k.shape
        Q = self.q_linear(q).view(B, S, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k_linear(k).view(B, S_k, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(v).view(B, S_k, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(B, S, self.d_model)
        output = self.out_linear(context)
        mean_attention = attention_weights.mean(dim=1)
        return output, mean_attention

class FF(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.mha = MHA(d_model, n_heads)
        self.ffn = FF(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, _ = self.mha(q=x, k=x, v=x, mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.masked_mha = MHA(d_model, n_heads)
        self.cross_mha = MHA(d_model, n_heads)
        self.ffn = FF(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        attn_out1, _ = self.masked_mha(q=x, k=x, v=x, mask=tgt_mask)
        x = self.norm1(x + self.dropout(attn_out1))
        attn_out2, cross_attn_weights = self.cross_mha(
            q=x, k=enc_output, v=enc_output, mask=src_mask
        )
        x = self.norm2(x + self.dropout(attn_out2))
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
        return x, cross_attn_weights

class TransformerBlock(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, n_heads,
                 d_ff, n_layers, max_len=10, dropout=0.1):
        super().__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pe = PE(d_model, max_len)
        self.encoder_layers = nn.ModuleList(
            [TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
             for _ in range(n_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model, n_heads, d_ff, dropout)
             for _ in range(n_layers)]
        )
        self.out_linear = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_out = self.pe(self.src_embedding(src))
        for layer in self.encoder_layers:
            enc_out = layer(enc_out, src_mask)
        dec_out = self.pe(self.tgt_embedding(tgt))
        for layer in self.decoder_layers:
            dec_out, cross_attn_weights = layer(dec_out, enc_out, src_mask, tgt_mask)
        logits = self.out_linear(dec_out)
        return logits, cross_attn_weights

# --- Mask helper functions ---
def make_src_mask(src, pad_token=0):
    return (src != pad_token).unsqueeze(1).unsqueeze(2)

def make_tgt_mask(tgt, pad_token=0):
    batch_size, tgt_len = tgt.shape
    tgt_pad_mask = (tgt != pad_token).unsqueeze(1).unsqueeze(2)
    causal_mask = (
        torch.triu(torch.ones((tgt_len, tgt_len), device=tgt.device), diagonal=1) == 0
    )
    tgt_mask = tgt_pad_mask & causal_mask.unsqueeze(0).unsqueeze(1)
    return tgt_mask

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for src_batch, tgt_batch in loader:
        src_batch = src_batch.to(device)
        tgt_batch = tgt_batch.to(device)

        tgt_input = tgt_batch[:, :-1]
        tgt_output = tgt_batch[:, 1:]

        src_mask = make_src_mask(src_batch)
        tgt_mask = make_tgt_mask(tgt_input)

        optimizer.zero_grad()
        logits, _ = model(src_batch, tgt_input, src_mask, tgt_mask)
        loss = criterion(
            logits.contiguous().view(-1, logits.size(-1)),
            tgt_output.contiguous().view(-1)
        )
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def main():
    device = torch.device("cpu")
    print("Training starting on CPU...")

    # Load dataset
    src_data_path = "C:/machine learning/intern/Seq-Seq-RNN/fra.txt"
    if not os.path.exists(src_data_path):
        print(f"Dataset not found at {src_data_path}")
        return

    print("Loading dataset from", src_data_path)
    df = pd.read_csv(
        src_data_path,
        sep="\t",
        names=["eng", "fra"],
        usecols=[0, 1],
        encoding="utf-8"
    )

    df["fra"] = df["fra"].astype(str).apply(normalize_string)
    df["eng"] = df["eng"].astype(str).apply(normalize_string)

    # Filter by sentence lengths
    df = df[
        (df["fra"].str.split(' ').str.len() < MAX_LENGTH) &
        (df["eng"].str.split(' ').str.len() < MAX_LENGTH)
    ]

    # Sample 5000 pairs for fast CPU training
    sample_size = min(5000, len(df))
    df = df.sample(n=sample_size, random_state=42)
    pairs = df[["fra", "eng"]].values.tolist()
    print(f"Loaded {len(pairs)} sentence pairs.")

    # Build vocabularies
    input_lang = Lang("fra")
    output_lang = Lang("eng")
    for pair in pairs:
        input_lang.add_sentence(pair[0])
        output_lang.add_sentence(pair[1])

    print(f"French Vocab: {input_lang.n_words} words, English Vocab: {output_lang.n_words} words")

    train_loader = prepare_data_loader(pairs, input_lang, output_lang, batch_size=64)

    # Initialize model
    D_MODEL = 256
    N_HEADS = 8
    D_FF = 512
    N_LAYERS = 3
    DROPOUT = 0.1

    model = TransformerBlock(
        src_vocab_size=input_lang.n_words,
        tgt_vocab_size=output_lang.n_words,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        d_ff=D_FF,
        n_layers=N_LAYERS,
        max_len=MAX_LENGTH,
        dropout=DROPOUT
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    optimizer = optim.Adam(model.parameters(), lr=0.0003, betas=(0.9, 0.98), eps=1e-9)

    epochs = 10
    print(f"Training model for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch}/{epochs} | Loss: {loss:.4f}")

    # Save checkpoint
    checkpoint_path = "transformer_fren_v1.pth"
    model_config = {
        'd_model': D_MODEL,
        'n_heads': N_HEADS,
        'd_ff': D_FF,
        'n_layers': N_LAYERS,
        'max_len': MAX_LENGTH,
        'dropout': DROPOUT
    }
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'input_lang_words': input_lang.word2index,
        'input_lang_idx': input_lang.index2word,
        'input_lang_n': input_lang.n_words,
        'output_lang_words': output_lang.word2index,
        'output_lang_idx': output_lang.index2word,
        'output_lang_n': output_lang.n_words,
        'config': model_config
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint successfully saved to {checkpoint_path}")

if __name__ == "__main__":
    main()
