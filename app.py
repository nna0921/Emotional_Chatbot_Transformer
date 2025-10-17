import os
import re
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


df = pd.read_csv("emotion-emotion_69k.csv")
df = df.dropna(axis=1, how='all')

def normalize_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[“”\"']", '"', text)
    text = re.sub(r"([?.!,])", r" \1 ", text)
    text = re.sub(r"[^a-zA-Z?.!,\"]+", " ", text)
    return text.strip()

inputs = []
targets = []

for _, row in df.iterrows():
    situation = normalize_text(row.get("Situation", ""))
    emotion = normalize_text(row.get("emotion", ""))
    cust = re.sub(r"customer\s*:", "", str(row.get("empathetic_dialogues", "")), flags=re.IGNORECASE)
    cust = re.sub(r"agent\s*:", "", cust, flags=re.IGNORECASE)
    cust = cust.replace("\\n", " ").strip()
    cust = normalize_text(cust)
    target = normalize_text(str(row.get("labels", "")))

    if not situation or not emotion or not cust or not target:
        continue

    x = f"Emotion: {emotion} | Situation: {situation} | Customer: {cust} Agent:"
    y = target

    inputs.append(x)
    targets.append(y)

data = pd.DataFrame({"input": inputs, "target": targets})

train_df, temp_df = train_test_split(data, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)


structural_tokens = ["Emotion:", "|", "Situation:", "Customer:", "Agent:"]

def tokenize(text, structural_tokens=structural_tokens):
    if not isinstance(text, str):
        return []
    text = text.lower()
    pattern_parts = [re.escape(tok.lower()) for tok in structural_tokens]
    full_pattern = "|".join(pattern_parts) + r"|\w+|[^\w\s]"
    return re.findall(full_pattern, text)

train_inputs = [tokenize(text) for text in train_df["input"]]
train_targets = [tokenize(text) for text in train_df["target"]]
all_train_tokens = [tok for seq in (train_inputs + train_targets) for tok in seq]

min_freq = 2
token_counts = Counter(all_train_tokens)
CORE_VOCAB_FILTERED = [
    tok for tok, freq in token_counts.items() 
    if freq >= min_freq and tok not in [t.lower() for t in structural_tokens]
]

special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>", "<sep>"]

raw_emotions = (
    train_df["input"]
    .str.extract(r"emotion:\s*(\w+)", flags=re.IGNORECASE)[0]
    .dropna()
    .unique()
    .tolist()
)
valid_emotions = [
    e for e in raw_emotions
    if isinstance(e, str) and e.isalpha() and e.lower() not in ["nan", "but", "we", "time", "i", "m", "a", "t"]
]
emotion_tokens = [f"<emotion_{e.lower()}>" for e in valid_emotions]

final_vocab = special_tokens + emotion_tokens + structural_tokens + CORE_VOCAB_FILTERED
word2idx = {word: idx for idx, word in enumerate(final_vocab)}
idx2word = {idx: word for word, idx in word2idx.items()}

PAD_IDX = word2idx["<pad>"]
SOS_IDX = word2idx["<bos>"]
EOS_IDX = word2idx["<eos>"]
UNK_IDX = word2idx["<unk>"]

def detokenize(tokens):
    if isinstance(tokens, str):
        tokens = tokens.split()
    text = " ".join(tokens)
    text = (text
            .replace("’", "'")
            .replace("‘", "'")
            .replace("“", '"')
            .replace("”", '"'))
    text = re.sub(r"\s+([?.!,])", r"\1", text)
    text = re.sub(r'(?<=\w)"(?=\w)', "'", text)
    text = re.sub(r"\s+'", "'", text)
    text = re.sub(r"'\s+", "'", text)
    text = re.sub(r'\s+"', '"', text)
    text = re.sub(r'"\s+', '"', text)
    text = re.sub(r'(?<=\b\w)"(?=\w\b)', "'", text)
    text = re.sub(r"([!?.,])\1{2,}", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    if text:
        text = text[0].upper() + text[1:]
    return text

def encode(tokens, max_len=64):
    ids = [word2idx.get(t, UNK_IDX) for t in tokens]
    ids = [SOS_IDX] + ids + [EOS_IDX]
    if len(ids) < max_len:
        ids += [PAD_IDX] * (max_len - len(ids))
    return torch.tensor(ids[:max_len])


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        B, Lq, _ = query.size()
        _, Lk, _ = key.size()
        q = self.q_proj(query).view(B, Lq, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(key).view(B, Lk, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(value).view(B, Lk, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, Lq, -1)
        return self.o_proj(out)

class PositionwiseFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, heads, dropout)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        attn = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, heads, dropout)
        self.ffn = PositionwiseFFN(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, tgt_mask, src_mask):
        x2 = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(x2))
        x2 = self.cross_attn(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + self.dropout(x2))
        x2 = self.ffn(x)
        x = self.norm3(x + self.dropout(x2))
        return x

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=256, heads=2, num_layers=2, d_ff=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.enc_layers = nn.ModuleList([EncoderLayer(d_model, heads, d_ff, dropout) for _ in range(num_layers)])
        self.dec_layers = nn.ModuleList([DecoderLayer(d_model, heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def encode(self, src, src_mask):
        x = self.dropout(self.pos_enc(self.embedding(src) * math.sqrt(self.d_model)))
        for layer in self.enc_layers:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, enc_out, tgt_mask, src_mask):
        x = self.dropout(self.pos_enc(self.embedding(tgt) * math.sqrt(self.d_model)))
        for layer in self.dec_layers:
            x = layer(x, enc_out, tgt_mask, src_mask)
        return self.fc_out(x)

    def make_src_mask(self, src):
        return (src != PAD_IDX).unsqueeze(1).unsqueeze(2)

    def make_tgt_mask(self, tgt):
        B, L = tgt.size()
        pad_mask = (tgt != PAD_IDX).unsqueeze(1).unsqueeze(3)
        causal = torch.tril(torch.ones(L, L, device=tgt.device)).bool()
        return pad_mask & causal


def greedy_decode(model, src, max_len=50, use_sampling=False, top_p=0.9, temperature=1.0):
    model.eval()
    if src.dim() == 1:
        src = src.unsqueeze(0)
    src = src.to(device)
    with torch.no_grad():
        src_mask = model.make_src_mask(src)
        enc_out = model.encode(src, src_mask)
    ys = torch.tensor([[SOS_IDX]], device=device)
    for _ in range(max_len):
        tgt_mask = model.make_tgt_mask(ys)
        with torch.no_grad():
            out = model.decode(ys, enc_out, tgt_mask, src_mask)
            logits = out[:, -1, :]
        if use_sampling:
            probs = torch.softmax(logits / temperature, dim=-1).squeeze(0)  # Squeeze for [V]
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cum_probs = torch.cumsum(sorted_probs, dim=-1)
            cut_off = (cum_probs > top_p).float()
            if cut_off.any():
                first_idx = torch.nonzero(cut_off)[0].item()
                sorted_probs = sorted_probs[:first_idx + 1]
                sorted_idx = sorted_idx[:first_idx + 1]
            sorted_probs /= sorted_probs.sum()
            sampled = torch.multinomial(sorted_probs, 1).item()
            next_tok = sorted_idx[sampled].item()
        else:
            next_tok = logits.argmax(-1).item()
        ys = torch.cat([ys, torch.tensor([[next_tok]], device=device)], dim=1)
        if next_tok == EOS_IDX:
            break
    return ys

def _get_ngrams(sequence, n):
    return {tuple(sequence[i:i + n]) for i in range(len(sequence) - n + 1)}

def beam_search(model, src, beam_size=4, max_len=50, length_penalty=0.7, no_repeat_ngram_size=3):
    model.eval()
    if src.dim() == 1:
        src = src.unsqueeze(0)
    src = src.to(device)
    with torch.no_grad():
        src_mask = model.make_src_mask(src)
        enc_out = model.encode(src, src_mask)

    Beam = [(0.0, [SOS_IDX], False)]
    for _ in range(max_len):
        all_candidates = []
        for score, seq, done in Beam:
            if done:
                all_candidates.append((score, seq, True))
                continue
            ys = torch.tensor([seq], device=device)
            tgt_mask = model.make_tgt_mask(ys)
            with torch.no_grad():
                out = model.decode(ys, enc_out, tgt_mask, src_mask)
                log_probs = F.log_softmax(out[:, -1, :], dim=-1)[0]
            topk_logp, topk_ids = torch.topk(log_probs, beam_size)
            for lp, tok in zip(topk_logp.tolist(), topk_ids.tolist()):
                new_seq = seq + [tok]
                if no_repeat_ngram_size > 0 and len(new_seq) > no_repeat_ngram_size:
                    ngram = tuple(new_seq[-no_repeat_ngram_size:])
                    if ngram in _get_ngrams(new_seq[:-no_repeat_ngram_size], no_repeat_ngram_size):
                        continue
                new_score = score + lp
                finished = (tok == EOS_IDX)
                all_candidates.append((new_score, new_seq, finished))
        Beam = sorted(all_candidates, key=lambda x: x[0], reverse=True)[:beam_size]
        if all(d for _, _, d in Beam):
            break
 
    normalized = sorted([(s / (len(seq) ** length_penalty), seq) for s, seq, _ in Beam], reverse=True)
    best_seq = normalized[0][1]
    return torch.tensor([best_seq], device=device)


@st.cache_resource
def load_model():
    vocab_size = len(final_vocab)
    model = TransformerModel(vocab_size=vocab_size).to(device)
    model_path = "/Users/mac/Desktop/gen_ai_22F3658_3684/best_transformer_chatbot (1).pt"
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        st.success("Model loaded successfully!")
        return model
    else:
        st.error(f"Model file {model_path} not found. Place it in the project folder.")
        return None

model = load_model()


st.markdown("""
    <style>
    /* Overall App */
    .main { 
        background-color: #f0f4f8; /* Soft blue-gray background */
        font-family: 'Helvetica', sans-serif;
    }
    
    /* Title */
    h1 { 
        text-align: center; 
        color: #4a90e2; /* Vibrant blue for title */
        font-family: 'Arial', sans-serif; 
        font-weight: bold;
        margin-bottom: 20px;
    }
    
    /* Chat Messages */
    .stChatMessage { 
        padding: 12px; 
        margin: 8px 0; 
        border-radius: 15px; 
        box-shadow: 0 2px 5px rgba(0,0,0,0.1); 
        transition: background-color 0.3s;
    }
    
    /* User Message (Right-aligned, Greenish) */
    [data-testid="stChatMessageUser"] { 
        background-color: #a8e6cf; /* Soft mint green */
        align-self: flex-end; 
        max-width: 70%; 
        margin-left: auto; 
        border: 1px solid #88d8b0; 
        color: #2e7d32; /* Dark green text */
    }
    
    /* Assistant Message (Left-aligned, Blueish) */
    [data-testid="stChatMessageAssistant"] { 
        background-color: #cde7ff; /* Soft light blue */
        align-self: flex-start; 
        max-width: 70%; 
        border: 1px solid #90caf9; 
        color: #1565c0; /* Deep blue text */
    }
    
    /* Inputs and Selectors */
    .stTextInput > div > div > input { 
        border-radius: 20px; 
        border: 1px solid #4a90e2; 
        background-color: #ffffff; 
        color: #333;
    }
    
    .stSelectbox > div > div { 
        border-radius: 10px; 
        border: 1px solid #4a90e2; 
        background-color: #e3f2fd; /* Light blue fill */
    }
    
    .stTextArea > div > div > textarea { 
        border-radius: 10px; 
        border: 1px solid #4a90e2; 
        background-color: #ffffff; 
    }
    
    /* Sidebar */
    .sidebar .sidebar-content { 
        background-color: #e1f5fe; /* Very light blue sidebar */
        border-right: 1px solid #90caf9;
    }
    
    /* Buttons */
    button[kind="primary"] {
        background-color: #4a90e2; /* Blue button */
        color: white;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #4a90e2 !important;
    }
    
    /* Warning */
    .stWarning {
        background-color: #fff3e0;
        border-color: #ffcc80;
        color: #e65100;
    }
    </style>
    """, unsafe_allow_html=True)


st.title("MoodBud")


with st.sidebar:
    st.header("Settings")
    emotions_list = [e.capitalize() for e in valid_emotions]
    selected_emotion = st.selectbox("Emotion", emotions_list)
    selected_method = st.selectbox("Decoding", ["Greedy", "Beam Search"])
    situation = st.text_area("Situation", height=100)
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()


if "messages" not in st.session_state:
    st.session_state.messages = []


chat_container = st.container()

with chat_container:
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(message["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(message["content"])


if prompt := st.chat_input("Type your message..."):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with chat_container:
        with st.chat_message("user"):
            st.markdown(prompt)

  
    if model and selected_emotion and situation:
        emotion_norm = normalize_text(selected_emotion)
        situation_norm = normalize_text(situation)
        customer_norm = normalize_text(prompt)
        input_str = f"Emotion: {emotion_norm} | Situation: {situation_norm} | Customer: {customer_norm} Agent:"
        tokens = tokenize(input_str)
        ids = encode(tokens).to(device)

        with st.spinner("Thinking..."):
            if selected_method == "Greedy":
                pred = greedy_decode(model, ids)
            else:
                pred = beam_search(model, ids, beam_size=3)
            decoded_tokens = [idx2word.get(t.item(), "") for t in pred[0] if t.item() not in [PAD_IDX, SOS_IDX, EOS_IDX]]
            response = detokenize(decoded_tokens)

       
        st.session_state.messages.append({"role": "assistant", "content": response})
        with chat_container:
            with st.chat_message("assistant"):
                st.markdown(response)
    else:
        st.warning("Set emotion, situation, and load model first.")


st.markdown("""
    <script>
    const chatContainer = window.parent.document.querySelector('.stContainer');
    if (chatContainer) chatContainer.scrollTop = chatContainer.scrollHeight;
    </script>
    """, unsafe_allow_html=True)

