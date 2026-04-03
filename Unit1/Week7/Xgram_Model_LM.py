# pip install gensim numpy pandas

import math, re
import numpy as np
import pandas as pd
from gensim.models import FastText
from gensim.utils import simple_preprocess
from pathlib import Path

TRAIN_CORPUS   = Path("Unit1/Week7/language_model_corpus_50000_sentences.txt")
CLOZE_SENTS    = Path("Unit1/Week7/cloze_corpus_500_sentences.txt")
CLOZE_ANSWERS  = Path("Unit1/Week7/cloze_corpus_500_answers.txt")

BLANK_TOKEN = "blank"   # simple_preprocess("[BLANK]") -> "blank"
token_re = re.compile(r"[A-Za-z0-9]+")

def ans_tok(s: str) -> str:
    m = token_re.search(s.strip())
    return m.group(0).lower() if m else ""

# ---------- 1) Train FastText ----------
# FastText learns subword vectors, so OOV words still get vectors.
sentences = [simple_preprocess(line) for line in TRAIN_CORPUS.open("r", encoding="utf-8", errors="ignore")]

ft = FastText(
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,          # skip-gram
    negative=10,
    epochs=10,
    workers=4,
    seed=42
)
ft.build_vocab(corpus_iterable=sentences)

ft.train(corpus_iterable=sentences, total_examples=ft.corpus_count, epochs=ft.epochs)

D = ft.vector_size

# ---------- 2) Build candidate vocabulary (include ALL gold answers) ----------
sent_lines = CLOZE_SENTS.read_text(encoding="utf-8", errors="ignore").splitlines()
ans_lines  = CLOZE_ANSWERS.read_text(encoding="utf-8", errors="ignore").splitlines()

gold = [ans_tok(a) for a in ans_lines]
base_vocab = list(ft.wv.index_to_key)
extra = sorted(set(gold) - set(base_vocab))
cand = base_vocab + extra

word2id = {w:i for i,w in enumerate(cand)}
V = len(cand)

# Precompute candidate vectors (FastText can produce vectors for extra/OOV)
cand_vecs = np.vstack([ft.wv.get_vector(w) for w in cand]).astype(np.float32)
cand_norms = np.linalg.norm(cand_vecs, axis=1, keepdims=True) + 1e-12
cand_vecs = cand_vecs / cand_norms  # normalize for cosine

def ctx_vec(tokens, blank_i):
    # context size=2: take 2 left + 2 right, concatenate (4*D)
    L = len(tokens)
    ctx = []
    for j in (blank_i-2, blank_i-1, blank_i+1, blank_i+2):
        if 0 <= j < L:
            ctx.append(ft.wv.get_vector(tokens[j]))
        else:
            ctx.append(np.zeros(D, dtype=np.float32))
    return np.concatenate(ctx).astype(np.float32)  # shape (4D,)

# ---------- 3) Feedforward network (1 hidden layer) ----------
# x -> ReLU -> softmax over candidate vocab
H = 256
rng = np.random.default_rng(42)
W1 = (rng.normal(0, 0.01, size=(4*D, H))).astype(np.float32)
b1 = np.zeros(H, dtype=np.float32)
W2 = (rng.normal(0, 0.01, size=(H, V))).astype(np.float32)
b2 = np.zeros(V, dtype=np.float32)

def relu(z): return np.maximum(z, 0)

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=1, keepdims=True)

# ---------- 4) Build training examples from TRAIN_CORPUS ----------
# Predict center word from ctx size=2
X, y = [], []
for line in TRAIN_CORPUS.open("r", encoding="utf-8", errors="ignore"):
    t = simple_preprocess(line)
    if not t: 
        continue
    for i, target in enumerate(t):
        if target not in word2id:
            continue
        X.append(ctx_vec(t, i))
        y.append(word2id[target])

X = np.vstack(X).astype(np.float32)
y = np.array(y, dtype=np.int64)

# ---------- 5) Train (mini-batch SGD) ----------
lr = 1e-2
epochs = 3
batch = 1024

N = X.shape[0]
for ep in range(epochs):
    perm = rng.permutation(N)
    for s in range(0, N, batch):
        idx = perm[s:s+batch]
        xb = X[idx]
        yb = y[idx]

        z1 = xb @ W1 + b1
        h  = relu(z1)
        z2 = h @ W2 + b2
        p  = softmax(z2)

        # gradient of cross-entropy
        p[np.arange(len(yb)), yb] -= 1.0
        p /= len(yb)

        dW2 = h.T @ p
        db2 = p.sum(axis=0)

        dh  = p @ W2.T
        dz1 = dh * (z1 > 0)

        dW1 = xb.T @ dz1
        db1 = dz1.sum(axis=0)

        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

# ---------- 6) Evaluate cloze + perplexity ----------
rows = []
logps = []
correct = 0

for i, (sline, aline) in enumerate(zip(sent_lines, ans_lines), start=1):
    t = simple_preprocess(sline)
    if BLANK_TOKEN not in t:
        raise ValueError(f"Missing BLANK on line {i}: {sline}")
    b = t.index(BLANK_TOKEN)

    x = ctx_vec(t, b).reshape(1, -1)
    h = relu(x @ W1 + b1)
    p = softmax(h @ W2 + b2)[0]

    pred = cand[int(np.argmax(p))]
    true = ans_tok(aline)

    p_true = float(p[word2id[true]]) if true in word2id else 1e-12
    logps.append(math.log(p_true))

    is_corr = int(pred == true)
    correct += is_corr

    top10_idx = np.argsort(-p)[:10]
    top10 = "; ".join([f"{cand[j]}:{p[j]:.4g}" for j in top10_idx])

    rows.append([i, sline, true, pred, p_true, is_corr, top10])

ppl = math.exp(-sum(logps)/len(logps))
acc = correct / len(rows)

df = pd.DataFrame(rows, columns=["line","sentence","true","pred","p_true","correct","top10"])
df.to_csv("cloze_fasttext_ffn_ctx2_predictions.csv", index=False)

print("Perplexity (FastText + FFN, ctx=2):", ppl)
print("Accuracy:", acc)
print("Wrote: cloze_fasttext_ffn_ctx2_predictions.csv")