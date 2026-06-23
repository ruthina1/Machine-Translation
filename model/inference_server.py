"""
Flask Inference Server for French → English Machine Translation.

Extracts model architecture + inference logic from Seq2Seq_Trans.ipynb
and serves predictions via a REST API.
"""

import os
import math
import re
import unicodedata

import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, jsonify
from flask_cors import CORS

# ──────────────────────────────────────────────
# Constants (must match training notebook)
# ──────────────────────────────────────────────
PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2
MAX_LENGTH = 10

# ──────────────────────────────────────────────
# Vocabulary
# ──────────────────────────────────────────────

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


# ──────────────────────────────────────────────
# Text normalization helpers
# ──────────────────────────────────────────────

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


# ──────────────────────────────────────────────
# Transformer Building Blocks
# ──────────────────────────────────────────────

class PE(nn.Module):
    """Positional Encoding."""
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
    """Multi-Head Attention."""
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
    """Position-wise Feed-Forward Network."""
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
    """Full Seq2Seq Transformer (Encoder + Decoder)."""
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


# ──────────────────────────────────────────────
# Mask helpers
# ──────────────────────────────────────────────

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


# ──────────────────────────────────────────────
# Checkpoint loading
# ──────────────────────────────────────────────

def load_checkpoint(filepath, device):
    """Rebuilds the entire translation pipeline from disc."""
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    config = checkpoint['config']

    in_lang = Lang("src")
    in_lang.word2index = checkpoint['input_lang_words']
    in_lang.index2word = checkpoint['input_lang_idx']
    in_lang.n_words = checkpoint['input_lang_n']

    out_lang = Lang("tgt")
    out_lang.word2index = checkpoint['output_lang_words']
    out_lang.index2word = checkpoint['output_lang_idx']
    out_lang.n_words = checkpoint['output_lang_n']

    restored_model = TransformerBlock(
        src_vocab_size=in_lang.n_words,
        tgt_vocab_size=out_lang.n_words,
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        d_ff=config['d_ff'],
        n_layers=config['n_layers'],
        max_len=config['max_len'],
        dropout=config['dropout']
    ).to(device)

    restored_model.load_state_dict(checkpoint['model_state_dict'])
    print(f"🔄 Checkpoint fully restored from {filepath}")

    return restored_model, in_lang, out_lang


# ──────────────────────────────────────────────
# Translation (autoregressive inference)
# ──────────────────────────────────────────────

def translate_sentence(model, sentence, input_lang, output_lang, device, max_len=10):
    model.eval()
    normalized = normalize_string(sentence)
    src_indices = pad_sequence(sentence_to_indices(input_lang, normalized), max_len)
    src_tensor = torch.tensor([src_indices], dtype=torch.long).to(device)
    src_mask = make_src_mask(src_tensor)
    tgt_indices = [SOS_TOKEN]

    for _ in range(max_len):
        tgt_tensor = torch.tensor([tgt_indices], dtype=torch.long).to(device)
        tgt_mask = make_tgt_mask(tgt_tensor)
        with torch.no_grad():
            logits, cross_attn_weights = model(src_tensor, tgt_tensor, src_mask, tgt_mask)
            next_token_logits = logits[0, -1, :]
            next_token = torch.argmax(next_token_logits).item()
        if next_token == EOS_TOKEN:
            break
        tgt_indices.append(next_token)

    translated_words = [
        output_lang.index2word.get(idx, "<UNK>") for idx in tgt_indices[1:]
    ]
    return " ".join(translated_words), cross_attn_weights


# ──────────────────────────────────────────────
# Flask Application
# ──────────────────────────────────────────────

app = Flask(__name__)
CORS(app)

# Global references filled at startup
model = None
input_lang = None
output_lang = None
device = None


@app.route('/predict', methods=['POST'])
def predict():
    """Translate French text to English."""
    data = request.get_json(force=True)
    text = data.get('text', '').strip()

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    try:
        translation, attn_weights = translate_sentence(
            model, text, input_lang, output_lang, device, max_len=MAX_LENGTH
        )
        # Convert attention weights to a list for JSON serialisation
        attn_list = attn_weights[0].cpu().detach().numpy().tolist()

        return jsonify({
            'translation': translation,
            'source': text,
            'attention_weights': attn_list,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
    })


# ──────────────────────────────────────────────
# Startup
# ──────────────────────────────────────────────

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Inference will run on: {device}")

    checkpoint_path = os.path.join(os.path.dirname(__file__), 'transformer_fren_v1.pth')

    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found at {checkpoint_path}")
        print("   Please run the training notebook first to generate the .pth file,")
        print("   then copy it into the model/ directory.")
        exit(1)

    model, input_lang, output_lang = load_checkpoint(checkpoint_path, device)
    print("✅ Model loaded and ready for inference")

    app.run(host='0.0.0.0', port=5000, debug=False)
