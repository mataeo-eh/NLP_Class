import torch
import torch.nn as nn
import csv
import os
import random
import re

# ─── Config ──────────────────────────────────────────────────────────────────
CORPUS_PATH = "Unit1/Week8/shakespeare_plays.csv"
MODEL_PATH  = "Unit1/Week8/rnn_autocomplete.pt"
EPOCHS        = 60
BATCH_SIZE    = 256
LEARNING_RATE = 0.001
WEIGHT_DECAY  = 0.01
PATIENCE      = 5
EMBED_DIM   = 64
HIDDEN_DIM  = 256
N_LAYERS    = 2
MAX_GEN_LEN = 20

# Special tokens
PAD   = '<PAD>'
START = '<S>'
END   = '</S>'

# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_words(path):
    """Extract all lowercase alphabetic words from the corpus."""
    words = []
    ext = os.path.splitext(path)[1].lower()

    if ext == '.csv':
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # skip header row
            for row in reader:
                if len(row) > 7:
                    tokens = re.findall(r'[a-z]+', row[7].lower())
                    words.extend(tokens)
    else:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                tokens = re.findall(r'[a-z]+', line.lower())
                words.extend(tokens)

    return words

def build_vocab(words):
    chars = sorted(set(''.join(words)))
    vocab = [PAD, START, END] + chars
    char2idx = {c: i for i, c in enumerate(vocab)}
    idx2char  = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char

def make_sequences(words, char2idx):
    """
    For each word, produce an (input, target) pair:
      input:  [<S>, c1, c2, ..., cn]
      target: [c1,  c2, ..., cn, </S>]
    The RNN learns: given characters so far, predict the next one.
    """
    seqs = []
    for word in words:
        if len(word) < 2:
            continue
        inp = [char2idx[START]] + [char2idx[c] for c in word]
        tgt = [char2idx[c] for c in word] + [char2idx[END]]
        seqs.append((inp, tgt))
    return seqs

def collate_batch(batch, pad_idx):
    inputs, targets = zip(*batch)
    max_len = max(len(x) for x in inputs)
    inp = torch.tensor(
        [x + [pad_idx] * (max_len - len(x)) for x in inputs], dtype=torch.long
    )
    tgt = torch.tensor(
        [x + [pad_idx] * (max_len - len(x)) for x in targets], dtype=torch.long
    )
    return inp, tgt

# ─── Model ────────────────────────────────────────────────────────────────────

class CharRNN(nn.Module):
    """
    Character-level RNN for word autocompletion.
    Uses a plain nn.RNN (not LSTM/GRU) as required by the exercise.
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers   = n_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # Plain RNN — tanh nonlinearity (PyTorch default)
        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            nonlinearity='tanh'
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        emb         = self.embedding(x)         # (B, T, embed_dim)
        out, hidden = self.rnn(emb, hidden)     # (B, T, hidden_dim)
        logits      = self.fc(out)              # (B, T, vocab_size)
        return logits, hidden

# ─── Training ─────────────────────────────────────────────────────────────────

def train(model, sequences, char2idx, device):
    # AdamW decouples weight decay from the gradient update, giving cleaner
    # regularization than plain Adam — better for a finite corpus like this.
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(ignore_index=char2idx[PAD])
    model.to(device)

    best_loss       = float('inf')
    best_state      = None
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        random.shuffle(sequences)
        total_loss = 0
        n_batches  = 0

        for i in range(0, len(sequences), BATCH_SIZE):
            batch = sequences[i : i + BATCH_SIZE]
            inp, tgt = collate_batch(batch, char2idx[PAD])
            inp, tgt = inp.to(device), tgt.to(device)

            optimizer.zero_grad()
            logits, _ = model(inp)
            loss = criterion(logits.view(-1, logits.size(-1)), tgt.view(-1))
            loss.backward()

            # Gradient clipping helps stabilize plain RNN training
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        avg_loss = total_loss / n_batches

        if avg_loss < best_loss:
            best_loss          = avg_loss
            best_state         = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve  = 0
            marker = " ✓"
        else:
            epochs_no_improve += 1
            marker = f" (no improvement {epochs_no_improve}/{PATIENCE})"

        print(f"Epoch {epoch:2d}/{EPOCHS} | Loss: {avg_loss:.4f}{marker}")

        if epochs_no_improve >= PATIENCE:
            print(f"\nEarly stopping. No improvement for {PATIENCE} epochs.")
            print(f"Restoring best model (loss: {best_loss:.4f}).")
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
            break

    if best_state and epochs_no_improve < PATIENCE:
        # Finished all epochs — still restore best in case last epoch wasn't best
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

# ─── Save / Load ──────────────────────────────────────────────────────────────

def save_model(model, char2idx, idx2char):
    torch.save(
        {
            'model_state': model.state_dict(),
            'char2idx':    char2idx,
            'idx2char':    idx2char,
            'config': {
                'vocab_size': len(char2idx),
                'embed_dim':  EMBED_DIM,
                'hidden_dim': HIDDEN_DIM,
                'n_layers':   N_LAYERS
            }
        },
        MODEL_PATH
    )
    print(f"Model saved → {MODEL_PATH}")

def load_model(device):
    checkpoint  = torch.load(MODEL_PATH, map_location=device)
    cfg         = checkpoint['config']
    char2idx    = checkpoint['char2idx']
    idx2char    = checkpoint['idx2char']
    model = CharRNN(
        vocab_size=cfg['vocab_size'],
        embed_dim=cfg['embed_dim'],
        hidden_dim=cfg['hidden_dim'],
        n_layers=cfg['n_layers']
    )
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    return model, char2idx, idx2char

# ─── Autocomplete ─────────────────────────────────────────────────────────────

def autocomplete(model, prefix, char2idx, idx2char, device):
    """
    Given a partial word prefix, predict the remaining characters
    one at a time until the end token or max length is reached.
    """
    model.eval()
    with torch.no_grad():
        indices = [char2idx[START]] + [char2idx.get(c, char2idx[PAD]) for c in prefix.lower()]
        x       = torch.tensor([indices], dtype=torch.long).to(device)
        logits, hidden = model(x)

        next_idx = logits[0, -1].argmax().item()
        result   = list(prefix.lower())

        for _ in range(MAX_GEN_LEN):
            next_char = idx2char[next_idx]
            if next_char == END:
                break
            result.append(next_char)

            x       = torch.tensor([[next_idx]], dtype=torch.long).to(device)
            logits, hidden = model(x, hidden)
            next_idx = logits[0, -1].argmax().item()

    return ''.join(result)

# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}\n")

    if os.path.exists(MODEL_PATH):
        print(f"Saved model found at '{MODEL_PATH}' — loading instead of retraining.")
        print("(Delete or move the file to retrain from scratch.)\n")
        model, char2idx, idx2char = load_model(device)
    else:
        print(f"No saved model found. Training on '{CORPUS_PATH}'...\n")
        words = load_words(CORPUS_PATH)
        print(f"  Total word tokens : {len(words):,}")
        print(f"  Unique words      : {len(set(words)):,}")

        char2idx, idx2char = build_vocab(words)
        print(f"  Character vocab   : {len(char2idx)} chars\n")

        sequences = make_sequences(words, char2idx)
        print(f"  Training sequences: {len(sequences):,}\n")

        model = CharRNN(
            vocab_size=len(char2idx),
            embed_dim=EMBED_DIM,
            hidden_dim=HIDDEN_DIM,
            n_layers=N_LAYERS
        )
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Model parameters  : {n_params:,}\n")
        print("─" * 40)

        train(model, sequences, char2idx, device)
        print()
        save_model(model, char2idx, idx2char)

    # Demo
    test_prefixes = ['thou', 'shal', 'lov', 'king', 'hon', 'hea', 'grac', 'nob']
    print("\n─── Autocomplete Demo ───")
    for p in test_prefixes:
        result = autocomplete(model, p, char2idx, idx2char, device)
        print(f"  '{p}' → '{result}'")

    # Interactive
    print("\n─── Interactive Autocomplete ───")
    print("Type a partial word and press Enter to see the RNN's completion.")
    print("(Press Ctrl+C to quit)\n")
    while True:
        try:
            prefix = input("Enter partial word: ").strip()
            if prefix:
                result = autocomplete(model, prefix, char2idx, idx2char, device)
                print(f"Autocompleted: '{result}'\n")
        except KeyboardInterrupt:
            print("\nExiting.")
            break
